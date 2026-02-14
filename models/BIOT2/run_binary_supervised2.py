import os
import argparse
import pickle
import time
from collections import deque
from queue import Queue
import threading

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

# import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.strategies import DDPStrategy
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pyhealth.metrics import binary_metrics_fn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score

from model import (
    # SPaRCNet,
    # ContraWR,
    # CNNTransformer,
    # FFCL,
    # STTransformer,
    BIOTClassifier,
)
from utils import TUABLoader, CHBMITLoader, PTBLoader, focal_loss, BCE

# import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torch.profiler import profile, ProfilerActivity, record_function

def set_seeds(args):
    pass

def prepare_dataloader(args):
    dataset = args.dataset
    subset = args.subset
    seed = args.seed

    if dataset == 'TUAB':
        return prepare_TUAB_dataloader(args)
    elif dataset == 'RANDOM':
        return prepare_RND_dataloader(args)

    pass

def prepare_model(args) -> torch.nn.Module:
    with torch.random.fork_rng():
        if args.model == "SPaRCNet":
            model = SPaRCNet(
                in_channels=args.in_channels,
                sample_length=int(args.sampling_rate * args.sample_length),
                n_classes=args.n_classes,
                block_layers=4,
                growth_rate=16,
                bn_size=16,
                drop_rate=0.5,
                conv_bias=True,
                batch_norm=True,
            )

        elif args.model == "ContraWR":
            model = ContraWR(
                in_channels=args.in_channels,
                n_classes=args.n_classes,
                fft=args.token_size,
                steps=args.hop_length // 5,
            )

        elif args.model == "CNNTransformer":
            model = CNNTransformer(
                in_channels=args.in_channels,
                n_classes=args.n_classes,
                fft=args.sampling_rate,
                steps=args.hop_length // 5,
                dropout=0.2,
                nhead=4,
                emb_size=256,
            )

        elif args.model == "FFCL":
            model = FFCL(
                in_channels=args.in_channels,
                n_classes=args.n_classes,
                fft=args.token_size,
                steps=args.hop_length // 5,
                sample_length=int(args.sampling_rate * args.sample_length),
                shrink_steps=20,
            )

        elif args.model == "STTransformer":
            model = STTransformer(
                emb_size=256,
                depth=4,
                n_classes=args.n_classes,
                channel_legnth=int(
                    args.sampling_rate * args.sample_length
                ),  # (sampling_rate * duration)
                n_channels=args.in_channels,
            )

        elif args.model == "BIOT":
            model = BIOTClassifier(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
                linear = args.linear, # modifications for tc_eeg
                top_k = args.top_k,
                tome_r = args.tome_r,
                flash=args.flash,
                tome_variant=args.tome_variant
            )
            if args.pretrain_model_path and (args.sampling_rate == 200):
                model.biot.load_state_dict(torch.load(args.pretrain_model_path))
                print(f"load pretrain model from {args.pretrain_model_path}")

        else:
            raise NotImplementedError
        
    return model

def calculate_metrics(y_hat, y_ground):
    if (
        sum(y_ground) * (len(y_ground) - sum(y_ground)) != 0
    ):  # to prevent all 0 or all 1 and raise the AUROC error
        threshold = np.sort(y_hat)[-int(np.sum(y_ground))]
        y_pred = np.empty_like(y_hat)
        y_pred[y_hat >= threshold] = 1
        y_pred[y_hat < threshold] = 0
        result = {
            "accuracy": accuracy_score(y_ground, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_ground, y_pred),
            "pr_auc": average_precision_score(y_ground, y_pred),
            "roc_auc": roc_auc_score(y_ground, y_pred),
        }
    else:
        result = {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "pr_auc": 0.0,
            "roc_auc": 0.0,
        }
    return result, threshold

# class LitModel_finetune(pl.LightningModule):
#     def __init__(self, args, model):
#         super().__init__()
#         self.model = model
#         self.threshold = 0.5
#         self.args = args

#     def training_step(self, batch, batch_idx):
#         X, y = batch
#         prob = self.model(X)
#         loss = BCE(prob, y)  # focal_loss(prob, y)
#         self.log("Train/train_loss", loss)
#         self.log("Mem/AllocMax", torch.cuda.max_memory_allocated() / 1024 / 1024)
#         self.log("Mem/ReservMax", torch.cuda.max_memory_reserved() / 1024 / 1024)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         X, y = batch
#         with torch.no_grad():
#             prob = self.model(X)
#             step_result = torch.sigmoid(prob).cpu().numpy()
#             step_gt = y.cpu().numpy()
#         return step_result, step_gt

#     def validation_epoch_end(self, val_step_outputs):
#         result = np.array([])
#         gt = np.array([])
#         for out in val_step_outputs:
#             result = np.append(result, out[0])
#             gt = np.append(gt, out[1])

#         if (
#             sum(gt) * (len(gt) - sum(gt)) != 0
#         ):  # to prevent all 0 or all 1 and raise the AUROC error
#             self.threshold = np.sort(result)[-int(np.sum(gt))]
#             result = binary_metrics_fn(
#                 gt,
#                 result,
#                 metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
#                 threshold=self.threshold,
#             )
#         else:
#             result = {
#                 "accuracy": 0.0,
#                 "balanced_accuracy": 0.0,
#                 "pr_auc": 0.0,
#                 "roc_auc": 0.0,
#             }
#         self.log("val_acc", result["accuracy"], sync_dist=True)
#         self.log("val_bacc", result["balanced_accuracy"], sync_dist=True)
#         self.log("val_pr_auc", result["pr_auc"], sync_dist=True)
#         self.log("val_auroc", result["roc_auc"], sync_dist=True)
#         print(result)

#     def test_step(self, batch, batch_idx):
#         X, y = batch
#         with torch.no_grad():
#             convScore = self.model(X)
#             step_result = torch.sigmoid(convScore).cpu().numpy()
#             step_gt = y.cpu().numpy()
#         return step_result, step_gt

#     def test_epoch_end(self, test_step_outputs):
#         result = np.array([])
#         gt = np.array([])
#         for out in test_step_outputs:
#             result = np.append(result, out[0])
#             gt = np.append(gt, out[1])
#         if (
#             sum(gt) * (len(gt) - sum(gt)) != 0
#         ):  # to prevent all 0 or all 1 and raise the AUROC error
#             result = binary_metrics_fn(
#                 gt,
#                 result,
#                 metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
#                 threshold=self.threshold,
#             )
#         else:
#             result = {
#                 "accuracy": 0.0,
#                 "balanced_accuracy": 0.0,
#                 "pr_auc": 0.0,
#                 "roc_auc": 0.0,
#             }
#         self.log("test_acc", result["accuracy"], sync_dist=True)
#         self.log("test_bacc", result["balanced_accuracy"], sync_dist=True)
#         self.log("test_pr_auc", result["pr_auc"], sync_dist=True)
#         self.log("test_auroc", result["roc_auc"], sync_dist=True)

#         return result

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(
#             self.model.parameters(),
#             lr=self.args.lr,
#             weight_decay=self.args.weight_decay,
#         )

#         return [optimizer]  # , [scheduler]


def prepare_TUAB_dataloader(args):
    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed"
    root = './datasets/TUAB/processed_' + args.subset

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_CHB_MIT_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/physionet.org/files/chbmit/1.0.0/clean_segments"

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "train"),
                     train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "test"),
                     test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_PTB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/WFDB/processed2"

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "train"),
                  train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

def prepare_RND_dataloader(args):
    # set random seed
    seed = args.seed
    # torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    # np.random.seed(seed)

    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples, n_channels, sampling_rate, sample_length):
            self.n_samples = n_samples
            self.data_shape = (n_channels, int(sampling_rate * sample_length))

        def __len__(self):
            return self.n_samples

        def __getitem__(self, index):
            X = torch.randn(self.data_shape, generator=g)
            y = torch.randint(0, 2, (1,), generator=g)#.float()
            return X, y

    train_dataset = RandomDataset(64, args.in_channels, args.sampling_rate, args.sample_length)
    val_dataset = RandomDataset(64, args.in_channels, args.sampling_rate, args.sample_length)
    test_dataset = RandomDataset(64, args.in_channels, args.sampling_rate, args.sample_length)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    return train_loader, test_loader, val_loader

class Context:
    epoch: int
    batch_count_train: int
    batch_count_eval : int
    losses: torch.Tensor
    preds:  torch.Tensor = None
    labels: torch.Tensor = None

    events_train: deque[tuple[int, torch.cuda.Event]]
    events_eval: deque[torch.cuda.Event]
    
    def __init__(self, epoch, bhs_train, bhs_eval):
        self.epoch = epoch
        self.batch_count_train = bhs_train
        self.batch_count_eval  = bhs_eval

        self.losses = torch.empty(bhs_train, dtype=torch.float, device='cpu', pin_memory=True, requires_grad=False)
        self.events_train = deque(maxlen=bhs_train)
        self.events_eval  = deque(maxlen=bhs_eval )


def train_loop_one_epoch(
        context : Context,
        model, loss_fn, optimizer, scheduler, train_dataloader : DataLoader, 
        stream_comp : torch.cuda.Stream, stream_cpin : torch.cuda.Stream, stream_cpout : torch.cuda.Stream, 
        **kwargs
        ):

    losses = context.losses
    events_train = context.events_train

    model.train()
    itor = iter(enumerate(train_dataloader))
    try:
        idx, (sample, label) = next(itor)
        with torch.cuda.stream(stream_cpin):
            sample = sample.to(stream_cpin.device, non_blocking=True)
            label = label.to(stream_cpin.device, non_blocking=True)
        stream_comp.wait_stream(stream_cpin)

        while True:
            with torch.cuda.stream(stream_comp):
                pred = model(sample)
                loss : torch.Tensor = loss_fn(pred, label)

            # transfer and log losses async
            stream_cpout.wait_stream(stream_comp)
            with torch.cuda.stream(stream_cpout):
                losses[idx].copy_(loss.detach(), non_blocking=True)
            events_train.append((idx, stream_cpout.record_event()))

            with torch.cuda.stream(stream_comp):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            try:
                sample_ : torch.Tensor
                label_  : torch.Tensor
                idx_, (sample_, label_) = next(itor)
                with torch.cuda.stream(stream_cpin):
                    sample_ = sample_.to(stream_cpin.device, non_blocking=True)
                    label_ = label_.to(stream_cpin.device, non_blocking=True)
                stream_comp.wait_stream(stream_cpin)

                idx, sample, label = idx_, sample_, label_
            except StopIteration:
                break
    except StopIteration:
        pass
    # for idx, (sample, label) in enumerate(train_dataloader):
        
    #     # copy data in dedicated stream
    #     with torch.cuda.stream(stream_cpin):
    #         sample = sample.to(stream_cpin.device, non_blocking=True)
    #         label = label.to(stream_cpin.device, non_blocking=True)

    #     # wait data to be copied
    #     stream_cpin.record_event().wait(stream_comp)

    #     # train the model
    #     with torch.cuda.stream(stream_comp):
    #         pred = model(sample)
    #         loss : torch.Tensor = loss_fn(pred, label)

    #     # transfer and log losses async
    #     stream_cpin.wait_stream(stream_comp)
    #     with torch.cuda.stream(stream_cpin):
    #         losses[idx].copy_(loss.detach(), non_blocking=True)
    #     events_train.append((idx, stream_cpin.record_event()))

    #     with torch.cuda.stream(stream_comp):
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()



def eval_loop_one_epoch(
        context : Context, 
        model, val_dataloader, eval_collate_fn, 
        stream_comp : torch.cuda.Stream, stream_cpin : torch.cuda.Stream, stream_cpout : torch.cuda.Stream, 
        **kwargs
        ):

    events_eval = context.events_eval
    eval_labels = context.labels
    eval_preds  = context.preds

    model.eval()
    with torch.no_grad():
        bsz = val_dataloader.batch_size
        itor = iter(enumerate(val_dataloader))
        try:
            idx, (sample, label) = next(itor)
            with torch.cuda.stream(stream_cpin):
                sample = sample.to(stream_cpin.device, non_blocking=True)

            stream_comp.wait_stream(stream_cpin)

            while True:
                with torch.cuda.stream(stream_comp):
                    pred : torch.Tensor = model(sample)
                    pred, label = eval_collate_fn(pred, label)

                if eval_preds is None:
                    eval_preds = torch.empty(len(val_dataloader.dataset), *pred.shape[1:], pin_memory=True, dtype=pred.dtype)
                if eval_labels is None:
                    eval_labels = torch.empty(len(val_dataloader.dataset), *label.shape[1:], pin_memory=True, dtype=label.dtype)

                event = stream_comp.record_event()
                events_eval.append(event)

                stream_cpout.wait_event(event)
                with torch.cuda.stream(stream_cpout):
                    eval_preds[idx * bsz : idx * bsz + pred.size(0)].copy_(pred, non_blocking=True)
                eval_labels[idx * bsz : idx * bsz + label.size(0)].copy_(label, False)

                try:
                    idx_, (sample_, label_) = next(itor)

                    with torch.cuda.stream(stream_cpin):
                        sample_ = sample_.to(stream_cpin.device, non_blocking=True)
                        stream_comp.wait_stream(stream_cpin)
                    
                    idx, sample, label = idx_, sample_, label_
                except StopIteration:
                    break
        except StopIteration:
            pass

    context.preds  = eval_preds
    context.labels = eval_labels

def train_loop(epochs, train_dataloader, val_dataloader, logger, stream_comp, stream_cpin, stream_cpout, **kwargs):
    bhs_train = len(train_dataloader)
    bhs_eval = len(val_dataloader)

    for epoch in range(epochs):
        context = Context(epoch, bhs_train, bhs_eval)
        pbar = tqdm(total=bhs_train+bhs_eval)

        events_timeit = [torch.cuda.Event(True) for _ in range(3)]
        finished = [threading.Event() for _ in range(2)]

        def monitor_loop():
            losses = context.losses
            events_train = context.events_train
            events_eval = context.events_eval

            while True:
                if len(events_train) > 0:
                    idx, event = events_train[0]
                    if event.query():
                        events_train.popleft()
                        loss = losses[idx].item()
                        logger.add_scalar('train/loss', loss, epoch * bhs_train + idx)
                        pbar.set_description(f"Epoch {epoch}/Train", False)
                        pbar.set_postfix({'loss': loss}, False)
                        pbar.update()

                if not finished[0].is_set():
                    time.sleep(0.005)
                    continue

                if len(events_eval) > 0:
                    event = events_eval[0]
                    if event.query():
                        events_eval.popleft()
                        pbar.set_description("Epoch {}/Eval".format(epoch), False)
                        pbar.update()

                if not finished[1].is_set():
                    time.sleep(0.005)
                    continue

                break

        monitor = threading.Thread(target=monitor_loop, name='cuda_event_monitor')
        monitor.start()

        try:
            events_timeit[0].record(stream_comp)
            train_loop_one_epoch(epoch = epoch, context=context, train_dataloader=train_dataloader, stream_comp=stream_comp, stream_cpin=stream_cpin, stream_cpout=stream_cpout, **kwargs)
            events_timeit[1].record(stream_comp)
            finished[0].set()
            eval_loop_one_epoch(epoch = epoch, context=context, val_dataloader=val_dataloader, stream_comp=stream_comp, stream_cpin=stream_cpin, stream_cpout=stream_cpout, **kwargs)
            events_timeit[2].record(stream_comp)

            events_timeit[1].synchronize()
            logger.add_scalar('train/elapse', events_timeit[0].elapsed_time(events_timeit[1]) / 1000, epoch)

            events_timeit[2].synchronize()
            logger.add_scalar('eval/elapse',  events_timeit[1].elapsed_time(events_timeit[2]) / 1000, epoch)
            logger.add_scalar('train/mem', torch.cuda.max_memory_allocated(), epoch)

            metrics, threshold = calculate_metrics(context.preds.numpy(), context.labels.numpy())
            for key, value in metrics.items():
                logger.add_scalar(f'eval/{key}', value, epoch)

            metrics2 = metrics.copy()
            metrics2['loss'] = context.losses.mean().item()
        finally:
            finished[0].set()
            finished[1].set()
            monitor.join()

        pbar.set_postfix(metrics2)
        pbar.close()
    logger.flush()

def test_loop(model, test_dataloader, test_collate_fn, stream_comp, stream_copy, logger, **kwargs):
    bhs_eval = len(test_dataloader)

    context = Context(0, 0, bhs_eval)
    pbar = tqdm(total=bhs_eval, desc="Test")

    events_timeit = [torch.cuda.Event(True) for _ in range(2)]
    finished = threading.Event()

    def monitor_loop():
        events = context.events_eval
        while True:
            while len(events) > 0:
                event = events[0]
                if event.query():
                    events.popleft()
                    pbar.update()

            if not finished.is_set():
                time.sleep(0.005)
                continue

            break

    monitor = threading.Thread(target=monitor_loop, name='cuda_event_monitor')
    monitor.start()

    try:
        events_timeit[0].record(stream_comp)
        eval_loop_one_epoch(context=context, model=model, val_dataloader=test_dataloader, eval_collate_fn=test_collate_fn, stream_comp=stream_comp, stream_cpin=stream_copy, **kwargs)
        events_timeit[1].record(stream_comp)

        events_timeit[1].synchronize()
        logger.add_scalar('test/elapse',  events_timeit[1].elapsed_time(events_timeit[0]) / 1000, 0)
        logger.add_scalar('test/mem', torch.cuda.max_memory_allocated(), 0)

        metrics = calculate_metrics(context.preds.numpy(), context.labels.numpy())
        for key, value in metrics.items():
            logger.add_scalar(f'test/{key}', value, 0)

        metrics2 = metrics.copy()
    finally:
        finished.set()
        monitor.join()

    pbar.set_postfix(metrics2)
    pbar.close()
    logger.flush()

def supervised(args):
    # get data loaders
    device = torch.device('cuda:0')

    # define streams
    stream_cpin = torch.cuda.Stream(device)
    stream_cpout = torch.cuda.Stream(device)
    stream_comp = torch.cuda.Stream(device)

    train_loader, test_loader, val_loader = prepare_dataloader(args)

    # define the model
    model = prepare_model(args)
    model.to(device)

    # define optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
    )

    loss_fn = BCE
    def eval_collate_fn(pred, label):
        pred = torch.sigmoid(pred)#.flatten(-2, -1)
        return pred, label

    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}-{args.seed}"
    logger = SummaryWriter(
        log_dir=f"{args.workspace}/logs/{args.log_dir}/{version}"
    )

    if not args.no_train:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
            train_loop(args.epochs, 
                    model = model, 
                    optimizer = optimizer,
                    loss_fn = loss_fn,
                    eval_collate_fn = eval_collate_fn,
                    train_dataloader = train_loader, 
                    val_dataloader = val_loader, 
                    logger = logger, 
                    stream_comp = stream_comp,
                    stream_cpin = stream_cpin,
                    stream_cpout = stream_cpout,
                    scheduler = None,
                    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=32, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUAB", help="dataset")
    parser.add_argument(
        "--model", type=str, default="SPaRCNet", help="which supervised model to use"
    )
    parser.add_argument(
        "--in_channels", type=int, default=16, help="number of input channels"
    )
    parser.add_argument(
        "--sample_length", type=float, default=10, help="length (s) of sample"
    )
    parser.add_argument(
        "--n_classes", type=int, default=1, help="number of output classes"
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=200, help="sampling rate (r)"
    )
    parser.add_argument("--token_size", type=int,
                        default=200, help="token size (t)")
    parser.add_argument(
        "--hop_length", type=int, default=100, help="token hop length (t - p)"
    )
    parser.add_argument(
        "--pretrain_model_path", type=str, default="", help="pretrained model path"
    )

    # modification made for tc_eeg
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--subset", type=str, default='')

    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--tome_r", type=int, nargs='+', default=[])
    parser.add_argument("--linear", action='store_true', default=False)
    parser.add_argument("--flash", action='store_true', default=False)

    parser.add_argument("--load_from_checkpoint", type=str, default=None)
    parser.add_argument("--no_train", action='store_true', default=False)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--profile", action='store_true', default=False)
    parser.add_argument("--workspace", type=str, required=True)
    # parser.add_argument("--rtl_tome", action='store_true', default=False)
    parser.add_argument("--tome_variant", type=str, default="")
    parser.add_argument("--cls_token", action='store_true', default=False)
    # end of modification

    args = parser.parse_args()
    print(args)

    supervised(args)
