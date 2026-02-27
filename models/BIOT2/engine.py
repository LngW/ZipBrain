import threading
import time
# from collections import deque
from typing import NamedTuple
from queue import Queue
from functools import partial
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import record_function

class __GlobalContext:
    __device : torch.device
    __s_compute : torch.cuda.Stream = None
    __s_mem_in  : torch.cuda.Stream = None
    __s_mem_out : torch.cuda.Stream = None
    __event_loop : threading.Thread = None
    __event_queue : Queue = None

    def __init__(self, device, queue, loop, log_dir, cp_dir):
        self.__device = device
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
        return iter((self.device, self.s_compute, self.s_mem_in, self.s_mem_out, self.__event_queue))

def create_global_context(device, log_dir, cp_dir):
    if not isinstance(device, torch.device):
        device = torch.device(device)

    queue = Queue()
    loop = __monitor(queue)
    
    return __GlobalContext(device, queue, loop, log_dir, cp_dir)

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

def __monitor_log_eval_batch(ctx : _EpochContext):
    pbar = ctx.pbar
    pbar.set_description("Epoch {}/Eval".format(ctx.epoch), False)
    pbar.update()

def __save_model(state_dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state_dict, path)

def __monitor_log_test_batch(ctx : _EpochContext):
    pbar = ctx.pbar
    pbar.set_description("Epoch {}/Test".format(ctx.epoch), False)
    pbar.update()

def __train_loop(
        model, loss_fn, optimizer : torch.optim.Optimizer, scheduler, dataloader : DataLoader, 
        ctx_global: __GlobalContext, ctx_epoch : _EpochContext, slots : _Slots,
        **kwargs
        ):
    
    # losses = ctx_epoch.losses
    losses = torch.empty(len(dataloader), dtype=torch.float, device='cpu', pin_memory=True, requires_grad=False)
    device, s_cmpt, s_cin, s_cout, queue = ctx_global

    e_in, e_cmpt = [torch.cuda.Event() for _ in range(2)]

    model.train()
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
                loss : torch.Tensor = loss_fn(model, pred, label)
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
        model : torch.nn.Module, dataloader, infer_post_fn, 
        ctx_global : __GlobalContext, ctx_epoch : _EpochContext, slots : _Slots,
        test_stage,
        **kwargs
        ):

    # events_eval = ctx_epoch.events_eval
    # eval_labels = ctx_epoch.labels
    # eval_preds  = ctx_epoch.preds
    eval_preds = None
    eval_labels = None

    device, s_cmpt, s_cin, s_cout, queue = ctx_global
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
                    pred, label = infer_post_fn(pred, label)

                    # computation completed
                    e_cmpt.record()
                    # events_eval.append(s_cmpt.record_event())
                    if test_stage:
                        queue.put((s_cmpt.record_event(), partial(__monitor_log_test_batch, ctx_epoch)))
                    else:
                        queue.put((s_cmpt.record_event(), partial(__monitor_log_eval_batch, ctx_epoch)))

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

    # ctx_epoch.preds  = eval_preds
    # ctx_epoch.labels = eval_labels

def train_loop(
        ctx_global : __GlobalContext,
        epochs : int, 
        model : torch.nn.Module,
        train_dataloader : DataLoader, 
        val_dataloader : DataLoader, 
        optimizer : torch.optim.Optimizer,
        loss_fn,
        infer_post_fn,
        metric_fn, 
        metric_comp_fn = None,
        scheduler = None,
        # **kwargs):
):
    
    device, s_cmpt, s_cin, s_cout, queue = ctx_global

    bhs_train = len(train_dataloader)
    bhs_eval = len(val_dataloader)

    slots = _Slots(2)

    class BestModel(NamedTuple):
        metrics : dict
        state_dict : dict
    best_model : BestModel = None

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
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_dataloader, 
            slots=slots,
            # **kwargs)
        )
        events_timeit[1].record(s_cmpt)

        preds, labels = __inference_loop(
            ctx_global=ctx_global,
            ctx_epoch=context, 
            model=model,
            dataloader=val_dataloader, 
            infer_post_fn=infer_post_fn,
            slots=slots,
            test_stage=False,
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
        metrics, threshold = metric_fn(preds.numpy(), labels.numpy())
        for key, value in metrics.items():
            ctx_global.log_epoch(f'eval/{key}', value, epoch)

        event = threading.Event()
        queue.put((None, event.set))
        event.wait()

        metrics2 = metrics.copy()
        metrics2['loss'] = loss
        pbar.write("Epoch\t{}:\t   ".format(context.epoch) + ", ".join(f"{k}: {v:.4f}" for k,v in metrics2.items()))
        pbar.close()

        state_dict = {
            "threshold": threshold,
            "model": model_state,
            "optimizer": optimizer_state,
            # "scheduler": sche
        }
        if metric_comp_fn is not None:
            better, fast_stop = metric_comp_fn(None if best_model is None else best_model.metrics, metrics2)
            if better:
                best_model = BestModel(metrics2, model_state)
                queue.put((None, partial(__save_model, state_dict, ctx_global.save_path("best.pt"))))        
        if fast_stop:
            break

    if state_dict is not None:
        __save_model(state_dict, ctx_global.save_path("last_epoch{}.pt".format(epoch)))

    ctx_global.logger.flush()

def test_loop(
        ctx_global : __GlobalContext,
        model, 
        test_dataloader, 
        infer_post_fn, 
        metric_fn, 
        logger, 
        **kwargs):

    bhs_eval = len(test_dataloader)
    device, s_cmpt, s_cin, s_cout, queue = ctx_global

    pbar = tqdm(total=bhs_eval, desc="Test")
    ctx_epoch = _EpochContext(0, pbar, logger, 0)

    events_timeit = [torch.cuda.Event(True) for _ in range(2)]

    events_timeit[0].record(s_cmpt)
    preds, labels = __inference_loop(
        ctx_global=ctx_global, ctx_epoch=ctx_epoch, 
        model=model, dataloader=test_dataloader, infer_post_fn=infer_post_fn, 
        **kwargs)
    events_timeit[1].record(s_cmpt)

    events_timeit[1].synchronize()
    logger.add_scalar('test/elapse',  events_timeit[1].elapsed_time(events_timeit[0]) / 1000, 0)
    logger.add_scalar('test/mem', torch.cuda.max_memory_allocated() / 1024 / 1024, 0)

    metrics = metric_fn(ctx_epoch.preds.numpy(), ctx_epoch.labels.numpy())
    for key, value in metrics.items():
        logger.add_scalar(f'test/{key}', value, 0)

    metrics2 = metrics.copy()

    pbar.set_postfix(metrics2)
    pbar.close()
    logger.flush()


# def train(
#         context, 
#         model, 
#         optimizer, 
#         scheduler, 
#         loss_fn, 
#         metric_fn, 
#         train_set, 
#         eval_set,
#         logger,
#         checkpoint
#         ):
#     pass
