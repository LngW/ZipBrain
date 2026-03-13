from dataclasses import dataclass
import threading
import time
from typing import NamedTuple, Callable, Any
from queue import Queue
from functools import partial
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import record_function

import numpy

class __GlobalContext:
    __device : torch.device
    __slots : '_Slots'
    __s_compute : torch.cuda.Stream = None
    __s_mem_in  : torch.cuda.Stream = None
    __s_mem_out : torch.cuda.Stream = None
    __event_loop : threading.Thread = None
    __event_queue : Queue = None

    def __init__(self, device, slots, queue, loop, log_dir, cp_dir):
        self.__device = device
        self.__slots = slots
        self.__event_queue = queue
        self.__event_loop = loop
        # self.__log_dir = log_dir
        self.__cp_dir = cp_dir

        self.logger = SummaryWriter(log_dir)
        self.global_step = 0

    @property
    def device(self):
        return self.__device

    @property
    def s_compute(self):
        if self.__s_compute is None:
            self.__s_compute = torch.cuda.default_stream(self.__device)
        return self.__s_compute

    @property
    def s_mem_in(self):
        if self.__s_mem_in is None:
            self.__s_mem_in = torch.cuda.Stream(self.__device)
        return self.__s_mem_in

    @property
    def s_mem_out(self):
        if self.__s_mem_out is None:
            self.__s_mem_out = torch.cuda.Stream(self.__device)
        return self.__s_mem_out
    
    def save_path(self, name):
        return self.__cp_dir / name
    
    def log_global_step(self, key, value):
        step = self.global_step
        self.global_step += 1

        self.logger.add_scalar(key, value, step)
    
    def log_global_step_(self, key, value):
        self.logger.add_scalar(key, value, self.global_step)
    
    def log_epoch(self, key, value, epoch):
        self.logger.add_scalar(key, value, epoch)

    def __iter__(self):
        return iter((self.device, self.__slots, self.s_compute, self.s_mem_in, self.s_mem_out, self.__event_queue))

@dataclass
class Hooks:
    calc_metric : Callable[[numpy.ndarray, numpy.ndarray], dict[str, Any]]
    ''' (pred, label) -> metrics '''

    calc_loss : Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor] = None
    '''(pred, label) -> loss'''

    compare_metric : Callable[[dict[str, Any], dict[str, Any]], bool] = None

    handle_post_train : Callable[[torch.nn.Module], None] = None
    '''decide which modules are freezed in train. this method is called after `model.train()`'''

    handle_post_infer_result : Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None
    '''(pred, label) -> (handled_pred, handled_label)'''

    compose_cp_custom_state : Callable[[dict[str, Any], dict[str, Any]], None] = None
    '''(result, metrics) -> None. Write custom state to the parameter `result`'''

    test_break : Callable[[bool, tuple[str, Any], tuple[str, Any]], bool] = None
    '''(better, old_metric, new_metric) -> is_break. Break the loop when return True'''

    schedule_step_batch : Callable[[torch.optim.lr_scheduler.LRScheduler], None] = None
    ''' (scheduler) -> None. Called after every batch finished. Step your scheduler in this method if necessary. '''

    schedule_step_epoch : Callable[[torch.optim.lr_scheduler.LRScheduler, dict[str, Any]], None] = None
    ''' (scheduler, metrics) -> None. Called after every epoch finished. Step your scheduler in this method if necessary. '''

def create_global_context(device, log_dir, cp_dir, slots = 2):
    if not isinstance(device, torch.device):
        device = torch.device(device)

    queue = Queue()
    loop = __monitor(queue)

    return __GlobalContext(device, _Slots(slots), queue, loop, log_dir, cp_dir)

class _EpochContext:
    epoch: int
    pbar : tqdm

    def __init__(self, epoch, pbar):
        self.epoch = epoch
        self.pbar = pbar

class _Slot:
    def __init__(self):
        self.reserved = False
        self.event = torch.cuda.Event()

    def reserve(self):
        if not self.event.query():
            self.event.synchronize()
        self.reserved = True

    def record(self, stream : torch.cuda.Stream):
        if self.reserved:
            stream.record_event(self.event)
            self.reserved = False

class _Slots:
    def __init__(self, max_slots):
        self.slots = [_Slot() for _ in range(max_slots)]
        self.idx = 0
    
    def next(self):
        slot = self.slots[self.idx]
        self.idx = (self.idx + 1) % len(self.slots)
        return slot
    
    def reserve_next(self):
        slot = self.next()
        slot.reserve()
        return slot

def __monitor(event_queue) -> threading.Thread:
    def __logic(event_queue : Queue):
        while True:
            e, processor = event_queue.get()
            while e is not None and not e.query():
                time.sleep(0.01)
            
            processor()

    thread = threading.Thread(target=__logic, args=(event_queue,), name='cuda_event_monitor', daemon=True)
    thread.start()

    return thread

def __monitor_log_train_batch(ctx_g : __GlobalContext, ctx_e : _EpochContext, loss: torch.Tensor):
    pbar = ctx_e.pbar

    loss = loss.item()

    ctx_g.log_global_step('train/loss', loss)
    pbar.set_description(f"Epoch {ctx_e.epoch}/Train", False)
    pbar.set_postfix({'loss': loss}, False)
    pbar.update()

def __monitor_log_eval_batch(ctx : _EpochContext, prefix : str):
    pbar = ctx.pbar
    pbar.set_description(prefix.format(ctx.epoch), False)
    pbar.update()

def __save_model(state_dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state_dict, path)

# def __monitor_log_test_batch(ctx : _EpochContext):
#     pbar = ctx.pbar
#     pbar.set_description("Epoch {}/Test".format(ctx.epoch), False)
#     pbar.update()

def __train_loop(
        model : torch.nn.Module, optimizer : torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler.LRScheduler, dataloader : DataLoader, 
        ctx_global: __GlobalContext, ctx_epoch : _EpochContext, hooks : Hooks,
        # handle_post_train = None,
        **kwargs
        ):
    
    # losses = ctx_epoch.losses
    losses = torch.empty(len(dataloader), dtype=torch.float, device='cpu', pin_memory=True, requires_grad=False)
    device, slots, s_cmpt, s_cin, s_cout, queue = ctx_global

    e_in, e_cmpt = [torch.cuda.Event() for _ in range(2)]

    model.train()
    if hooks.handle_post_train is not None:
        hooks.handle_post_train(model)
    # handle_post_train(model)
    itor = iter(enumerate(dataloader))
    try:
        idx, (sample, label) = next(itor)

        slot = slots.reserve_next()
        with torch.cuda.stream(s_cin):
            sample = sample.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            # now s_cmpt should wait util data transfered in
            e_in.record()

        while True:
            with torch.cuda.stream(s_cmpt):
                e_in.wait()
                pred = model(sample)
                loss : torch.Tensor = hooks.calc_loss(model, pred, label)
                # now s_cout should wait util computation finish
                e_cmpt.record()

            # transfer and log losses async
            with torch.cuda.stream(s_cout):
                e_cmpt.wait()
                losses[idx].copy_(loss.detach(), non_blocking=True)
                queue.put(
                    (s_cout.record_event(), partial(__monitor_log_train_batch, ctx_global, ctx_epoch, losses[idx]))
                )
                # events_train.append((idx, s_cout.record_event()))

            # continue backward pass
            with torch.cuda.stream(s_cmpt):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and hooks.schedule_step_batch is not None:
                    hooks.schedule_step_batch(scheduler)

                # we can deem this slot is free after the backward completed
                slot.record(s_cmpt)

            try:
                sample_ : torch.Tensor
                label_  : torch.Tensor
                idx_, (sample_, label_) = next(itor)

                slot = slots.reserve_next()
                with torch.cuda.stream(s_cin):
                    sample_ = sample_.to(device, non_blocking=True)
                    label_ = label_.to(device, non_blocking=True)
                    e_in.record()

                idx, sample, label = idx_, sample_, label_
            except StopIteration:
                break
    except StopIteration:
        pass

    s_cout.wait_stream(s_cmpt) # in case backward is not finished
    # dump state_dict to cpu, so we can dump the state_dict to disk
    with torch.cuda.stream(s_cout):
        model_state_dict = model.state_dict().copy()
        for key in model_state_dict.keys():
            model_state_dict[key] = model_state_dict[key].to(device='cpu', non_blocking=True)
        
        optimizer_state_dict = optimizer.state_dict().copy()
        
        def _cast_to_cpu(v):
            if isinstance(v, torch.Tensor):
                return v.to(device='cpu', non_blocking=True)
            
            elif isinstance(v, dict):
                tmp = {}
                for k, v_ in v.items():
                    tmp[k] = _cast_to_cpu(v_)
                return tmp
            else:
                return v

        for key in optimizer_state_dict.keys():
            optimizer_state_dict[key] = _cast_to_cpu(optimizer_state_dict[key])

        # state_dict = {}
        # state_dict['model'] = model_state_dict
        # state_dict['optimizer'] = optimizer_state_dict

    dump_event = s_cout.record_event()
    
    return dump_event, model_state_dict, optimizer_state_dict, losses

def __inference_loop(
        model : torch.nn.Module, dataloader,
        ctx_global : __GlobalContext, ctx_epoch : _EpochContext, hooks : Hooks,
        # test_stage, 
        # valid=False,
        prefix : str,
        **kwargs
        ):

    # events_eval = ctx_epoch.events_eval
    # eval_labels = ctx_epoch.labels
    # eval_preds  = ctx_epoch.preds
    eval_preds = None
    eval_labels = None

    device, slots, s_cmpt, s_cin, s_cout, queue = ctx_global
    e_in, e_cmpt = [torch.cuda.Event() for _ in range(2)]

    model.eval()
    with torch.no_grad():

        bsz = dataloader.batch_size
        itor = iter(enumerate(dataloader))
        try:
            idx, (sample, label) = next(itor)
            slot = slots.reserve_next()
            with torch.cuda.stream(s_cin):
                sample = sample.to(device, non_blocking=True)
                e_in.record()

            while True:
                with torch.cuda.stream(s_cmpt):
                    e_in.wait()
                    pred : torch.Tensor = model(sample)
                    # we can move in next batch of data now
                    slot.record(s_cmpt)

                    # continue computation
                    if hooks.handle_post_infer_result is not None:
                        pred, label = hooks.handle_post_infer_result(pred, label)
                    # pred, label = infer_post_fn(pred, label)

                    # computation completed
                    e_cmpt.record()
                    # events_eval.append(s_cmpt.record_event())
                    # if test_stage:
                    #     queue.put((s_cmpt.record_event(), partial(__monitor_log_test_batch, ctx_epoch)))
                    # else:
                    queue.put((s_cmpt.record_event(), partial(__monitor_log_eval_batch, ctx_epoch, prefix)))

                if eval_preds is None:
                    eval_preds = torch.empty(len(dataloader.dataset), *pred.shape[1:], pin_memory=True, dtype=pred.dtype, device='cpu')
                if eval_labels is None:
                    eval_labels = torch.empty(len(dataloader.dataset), *label.shape[1:], pin_memory=True, dtype=label.dtype, device='cpu')
                eval_labels[idx * bsz : idx * bsz + label.size(0)].copy_(label, False)

                with torch.cuda.stream(s_cout):
                    e_cmpt.wait()
                    eval_preds[idx * bsz : idx * bsz + pred.size(0)].copy_(pred, non_blocking=True)

                try:
                    idx_, (sample_, label_) = next(itor)
                    slot = slots.reserve_next()
                    with torch.cuda.stream(s_cin):
                        sample_ = sample_.to(s_cin.device, non_blocking=True)
                        e_in.record()
                    
                    idx, sample, label = idx_, sample_, label_
                except StopIteration:
                    break
        except StopIteration:
            pass

    return eval_preds, eval_labels

def _train_loop(
        ctx_global : __GlobalContext,
        epochs : int, 
        model : torch.nn.Module,
        train_dataloader : DataLoader, 
        val_dataloader : DataLoader, 
        optimizer : torch.optim.Optimizer,
        scheduler : torch.optim.lr_scheduler.LRScheduler,
        hooks : Hooks,
        **kwargs,
    ):
    
    device, slots, s_cmpt, s_cin, s_cout, queue = ctx_global

    bhs_train = len(train_dataloader)
    bhs_eval = len(val_dataloader)

    class BestModel(NamedTuple):
        metrics : dict
        state_dict : dict
    best_model : BestModel = None

    with torch.random.fork_rng():
        iter(train_dataloader)
        iter(val_dataloader)

    state_dict = None
    for epoch in range(epochs):
        pbar = tqdm(total=bhs_train+bhs_eval, leave=False)
        context = _EpochContext(epoch, pbar)

        events_timeit = [torch.cuda.Event(True) for _ in range(3)]

        events_timeit[0].record(s_cmpt)
        dump_event, model_state, optimizer_state, losses = __train_loop(
            ctx_global=ctx_global,
            ctx_epoch=context, 
            model=model,
            hooks=hooks,
            # loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_dataloader, 
            **kwargs,
            # **kwargs)
        )
        events_timeit[1].record(s_cmpt)

        preds, labels = __inference_loop(
            ctx_global=ctx_global,
            ctx_epoch=context, 
            model=model,
            dataloader=val_dataloader, 
            hooks=hooks,
            prefix="Epoch {}/Eval"
            # **kwargs)
        )
        events_timeit[2].record(s_cmpt)

        # now commands are all issued, wait commands to be done
        events_timeit[1].synchronize()
        loss = losses.mean().item()
        ctx_global.log_epoch('train/elapse', events_timeit[0].elapsed_time(events_timeit[1]) / 1000, epoch)
        ctx_global.log_epoch('train/epoch_loss', loss, epoch)

        # wait eval loop to be done
        events_timeit[2].synchronize()
        ctx_global.log_epoch('eval/elapse',  events_timeit[1].elapsed_time(events_timeit[2]) / 1000, epoch)
        ctx_global.log_epoch('train/mem', torch.cuda.max_memory_allocated() / 1024 / 1024, epoch)

        # wait preds and models to be copied
        s_cout.synchronize()
        # calculate metrics for each epoch
        metrics = hooks.calc_metric(preds.numpy(), labels.numpy())
        for key, value in metrics.items():
            ctx_global.log_epoch(f'eval/{key}', value, epoch)

        event = threading.Event()
        queue.put((None, event.set))
        event.wait()

        metrics2 = metrics.copy()
        metrics2['loss'] = loss
        if scheduler is not None and hooks.schedule_step_epoch is not None:
            last_lr = scheduler.get_last_lr()[0]
            hooks.schedule_step_epoch(scheduler, metrics2)
            metrics2['lr'] = last_lr

        pbar.write("Epoch{:>4}:\t".format(context.epoch) + (" " * 4).join(f"{k}={v:.6f}" for k,v in metrics2.items()))
        pbar.close()


        custom_state = {}
        if hooks.compose_cp_custom_state is not None:
            hooks.compose_cp_custom_state(custom_state, metrics2)

        state_dict = {
            "model": model_state,
            "optimizer": optimizer_state,
            "custom": custom_state,
        }
        better = best_model is None or hooks.compare_metric(best_model.metrics, metrics2)
        old_state = None if best_model is None else best_model.metrics 
        if better:
            best_model = BestModel(metrics2, model_state)
            queue.put((None, partial(__save_model, state_dict, ctx_global.save_path("best.pt"))))

        if hooks.test_break is not None and hooks.test_break(better, old_state, metrics2):
            break

    if state_dict is not None:
        __save_model(state_dict, ctx_global.save_path("last_epoch{}.pt".format(epoch)))

    ctx_global.logger.flush()

def _inference_loop(
        ctx_global : __GlobalContext,
        model : torch.nn.Module, 
        test_dataloader : DataLoader, 
        hooks : Hooks,
        prefix : str = 'Test',
        **kwargs):

    bhs_eval = len(test_dataloader)
    device, slots, s_cmpt, s_cin, s_cout, queue = ctx_global

    pbar = tqdm(total=bhs_eval, desc=prefix, leave=False)
    ctx_epoch = _EpochContext(0, pbar)
    # slots = _Slots(2)

    events_timeit = [torch.cuda.Event(True) for _ in range(2)]

    events_timeit[0].record(s_cmpt)
    preds, labels = __inference_loop(
        ctx_global=ctx_global, ctx_epoch=ctx_epoch, 
        model=model, dataloader=test_dataloader, hooks=hooks,
        prefix = prefix,
        # test_stage=True,
        **kwargs)
    events_timeit[1].record(s_cmpt)

    prefix_lower = prefix.lower()
    events_timeit[1].synchronize()
    ctx_global.log_epoch(prefix_lower + '/elapse',  events_timeit[0].elapsed_time(events_timeit[1]) / 1000, 0)
    ctx_global.log_epoch(prefix_lower + '/mem', torch.cuda.max_memory_allocated() / 1024 / 1024, 0)

    metrics = hooks.calc_metric(preds.numpy(), labels.numpy())
    for key, value in metrics.items():
        ctx_global.log_epoch(f'{prefix_lower}/{key}', value, 0)

    metrics2 = metrics.copy()

    pbar.write(prefix + ":\t\t" + (" " * 4).join(f"{k}={v:.6f}" for k,v in metrics2.items()))
    pbar.close()
    ctx_global.logger.flush()

def valid_loop(
        ctx_global : __GlobalContext,
        # hooks: Hooks,
        model, 
        test_dataloader, 
        infer_post_fn, 
        metric_fn, 
        **kwargs):
    
    bhs_eval = len(test_dataloader)
    device, slots, s_cmpt, s_cin, s_cout, queue = ctx_global

    pbar = tqdm(total=bhs_eval, desc="Test", leave=False)
    ctx_epoch = _EpochContext(0, pbar)
    # slots = _Slots(2)

    events_timeit = [torch.cuda.Event(True) for _ in range(2)]

    events_timeit[0].record(s_cmpt)
    preds, labels = __inference_loop(
        ctx_global=ctx_global, ctx_epoch=ctx_epoch, 
        model=model, dataloader=test_dataloader, infer_post_fn=infer_post_fn, 
        # slots=slots, 
        test_stage=True, 
        valid=True,
        **kwargs)
    events_timeit[1].record(s_cmpt)

    events_timeit[1].synchronize()
    ctx_global.log_epoch('test/elapse',  events_timeit[1].elapsed_time(events_timeit[0]) / 1000, 0)
    ctx_global.log_epoch('test/mem', torch.cuda.max_memory_allocated() / 1024 / 1024, 0)

    metrics, _ = metric_fn(preds.numpy(), labels.numpy())
    for key, value in metrics.items():
        ctx_global.log_epoch(f'test/{key}', value, 0)

    metrics2 = metrics.copy()

    pbar.write("Test:\t\t" + (" " * 4).join(f"{k}={v:.6f}" for k,v in metrics2.items()))
    # pbar.set_postfix(metrics2)
    pbar.close()
    ctx_global.logger.flush()

def train(
        ctx_global : __GlobalContext,
        epochs : int, 
        model : torch.nn.Module,
        train_dataloader : DataLoader, 
        val_dataloader : DataLoader, 
        optimizer : torch.optim.Optimizer,
        hooks : Hooks,
        scheduler = None,
        **kwargs,
):
    if hooks.calc_loss is None or hooks.compare_metric is None:
        raise NotImplementedError("both calc_loss and compare_metric are required in training, but they are not defined in hooks")
    
    _train_loop(
        ctx_global = ctx_global, 
        epochs = epochs, 
        model = model, 
        train_dataloader = train_dataloader, 
        val_dataloader = val_dataloader, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        hooks = hooks, 
        **kwargs
    )

def test(
        ctx_global : __GlobalContext,
        model : torch.nn.Module, 
        test_dataloader : DataLoader, 
        hooks : Hooks,
        **kwargs
):
    _inference_loop(
        ctx_global=ctx_global, model=model, test_dataloader=test_dataloader, hooks=hooks, prefix = 'Test', **kwargs
    )

def valid(
        ctx_global : __GlobalContext,
        model : torch.nn.Module, 
        test_dataloader : DataLoader, 
        hooks : Hooks,
        **kwargs
):  
    from torch.utils.data import Dataset
    class LimitedDataset(Dataset):
        def __init__(self, upper, limit : int):
            super().__init__()
            self.upper = upper
            self.limit = limit
        
        def __len__(self):
            return min(self.limit, len(self.upper))

        def __getitem__(self, index):
            return self.upper.__getitem__(index)

    dataset = LimitedDataset(test_dataloader.dataset, test_dataloader.batch_size)
    loader = DataLoader(
        dataset=dataset,
        batch_size=test_dataloader.batch_size,
        num_workers=test_dataloader.num_workers,
        persistent_workers=test_dataloader.persistent_workers,
        pin_memory=test_dataloader.pin_memory,
    )
    
    _inference_loop(
        ctx_global=ctx_global, model=model, test_dataloader=loader, hooks=hooks, prefix = 'Valid', **kwargs
    )
