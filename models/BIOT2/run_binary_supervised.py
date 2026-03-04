import os
import argparse
import pickle
import time
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pyhealth.metrics import binary_metrics_fn

from model import (
    # SPaRCNet,
    # ContraWR,
    # CNNTransformer,
    # FFCL,
    # STTransformer,
    BIOTClassifier,
)
from utils import TUABLoader, CHBMITLoader, PTBLoader, focal_loss, BCE

from torch.profiler import profile, ProfilerActivity, record_function

class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.model : torch.nn.Module = model
        self.threshold = 0.5
        self.args = args

    def training_step(self, batch, batch_idx):
        X, y = batch
        prob = self.model(X)
        loss = BCE(prob, y)  # focal_loss(prob, y)
        for module in self.model.modules():
            if hasattr(module, 'compression_loss'):
                loss = loss + module.compression_loss
        self.log("Train/train_loss", loss)
        self.log("Mem/AllocMax", torch.cuda.max_memory_allocated() / 1024 / 1024)
        self.log("Mem/ReservMax", torch.cuda.max_memory_reserved() / 1024 / 1024)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            prob = self.model(X)
            step_result = torch.sigmoid(prob).cpu().numpy()
            step_gt = y.cpu().numpy()
        return step_result, step_gt

    def validation_epoch_end(self, val_step_outputs):
        result = np.array([])
        gt = np.array([])
        for out in val_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])

        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            self.threshold = np.sort(result)[-int(np.sum(gt))]
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("val_auroc", result["roc_auc"], sync_dist=True)

        result_ = 'Epoch {:>3d}:\t'.format(self.current_epoch) + (' ' * 4).join(['{}={:.6f}'.format(k, v) for k,v in result.items()])
        self.print(result_)

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = torch.sigmoid(convScore).cpu().numpy()
            step_gt = y.cpu().numpy()
        return step_result, step_gt

    def test_epoch_end(self, test_step_outputs):
        result = np.array([])
        gt = np.array([])
        for out in test_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("test_auroc", result["roc_auc"], sync_dist=True)

        return result

    def configure_optimizers(self):
        # assert isinstance(self.model, torch.nn.Module)
        # params = self.model.named_parameters()
        # base_biot = [p for name, p in params if 'transformer' not in name]
        # transformers = [p for name, p in params if 'transformer' in name]
        # groups = [
        #     {'params': base_biot},
        #     # {'params': transformers, 'lr': self.args.lr * 0.2},
        # ]
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            # groups,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10, 20, 40], gamma=1/1.41)

        return [optimizer]  #, [scheduler]
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(self.current_epoch)


def prepare_TUAB_dataloader(args):
    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed"

    if args.subset is not None and len(args.subset) > 0:
        root = './datasets/TUAB/processed_' + args.subset
    else:
        root = './datasets/TUAB/processed'

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
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
            y = torch.randint(0, 2, (1,), generator=g).float()
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

def supervised(args):

    cwd = Path(*args.workspace.split('/'))
    save_dir = cwd / 'logs'
    prof_dir = cwd / 'profiles'

    # get data loaders
    if args.dataset == "TUAB":
        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)
    elif args.dataset == "RANDOM":
        train_loader, test_loader, val_loader = prepare_RND_dataloader(args)
    else:
        raise NotImplementedError

    # define the model
    with torch.random.fork_rng():
        # if args.model == "SPaRCNet":
        #     model = SPaRCNet(
        #         in_channels=args.in_channels,
        #         sample_length=int(args.sampling_rate * args.sample_length),
        #         n_classes=args.n_classes,
        #         block_layers=4,
        #         growth_rate=16,
        #         bn_size=16,
        #         drop_rate=0.5,
        #         conv_bias=True,
        #         batch_norm=True,
        #     )

        # elif args.model == "ContraWR":
        #     model = ContraWR(
        #         in_channels=args.in_channels,
        #         n_classes=args.n_classes,
        #         fft=args.token_size,
        #         steps=args.hop_length // 5,
        #     )

        # elif args.model == "CNNTransformer":
        #     model = CNNTransformer(
        #         in_channels=args.in_channels,
        #         n_classes=args.n_classes,
        #         fft=args.sampling_rate,
        #         steps=args.hop_length // 5,
        #         dropout=0.2,
        #         nhead=4,
        #         emb_size=256,
        #     )

        # elif args.model == "FFCL":
        #     model = FFCL(
        #         in_channels=args.in_channels,
        #         n_classes=args.n_classes,
        #         fft=args.token_size,
        #         steps=args.hop_length // 5,
        #         sample_length=int(args.sampling_rate * args.sample_length),
        #         shrink_steps=20,
        #     )

        # elif args.model == "STTransformer":
        #     model = STTransformer(
        #         emb_size=256,
        #         depth=4,
        #         n_classes=args.n_classes,
        #         channel_legnth=int(
        #             args.sampling_rate * args.sample_length
        #         ),  # (sampling_rate * duration)
        #         n_channels=args.in_channels,
        #     )

        if args.model == "BIOT":
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
                tome_variant=args.tome_variant,
                cls_token = args.cls_token,
                merge_in_attn=args.merge_in_attn,
            )
            if args.pretrain_model_path and (args.sampling_rate == 200):
                model.biot.load_state_dict(torch.load(args.pretrain_model_path))
                print(f"load pretrain model from {args.pretrain_model_path}")

        else:
            raise NotImplementedError

    if args.load_from_checkpoint is not None:
        # state_dict = torch.load(args.load_from_checkpoint, weights_only=False)
        # print(state_dict)
        lightning_model = LitModel_finetune.load_from_checkpoint(args.load_from_checkpoint, args=args, model = model)
    else:
        lightning_model = LitModel_finetune(args, model)

    # print(model)

    # logger and callbacks
    # version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}"
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}-{args.seed}"
    logger = TensorBoardLogger(
        save_dir=save_dir, # modification for tc_eeg
        version=version,
        name=args.log_dir, # modification for tc_eeg
    )
    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=5, verbose=False, mode="max"
    )

    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        # strategy=DDPStrategy(find_unused_parameters=False),
        auto_select_gpus=True,
        benchmark=False,
        deterministic=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback],
        log_every_n_steps=1,
    )

    # train the model
    def profiling_func():
        if not args.no_train:
            with record_function('train'):
                trainer.fit(
                    lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
                )

        if args.test:
            # test the model
            with record_function('test'):
                pretrain_result = trainer.test(
                    model=lightning_model, ckpt_path=(None if args.no_train else "best"), dataloaders=test_loader
                )[0]

            print(pretrain_result)

    if args.profile:
        # prof_dir = f'./{args.workspace}/profiles'
        torch.cuda.memory._record_memory_history()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
            profiling_func()

        prof_dir.mkdir(exist_ok=True)
        prof.export_chrome_trace(str(prof_dir / f'{args.log_dir}_{time.time_ns() % (10 ** 8)}.json'))
        torch.cuda.memory._dump_snapshot(str(prof_dir / "dump_snapshot.pickle"))
    else:
        profiling_func()
    logger.finalize('')


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
    parser.add_argument("--tome_variant", type=str, nargs='+', default=[])
    parser.add_argument("--cls_token", action='store_true', default=False)
    parser.add_argument("--merge_in_attn", action='store_true', default=False)
    # end of modification

    args = parser.parse_args()
    print(args)

    supervised(args)
