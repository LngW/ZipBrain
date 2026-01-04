import os
import argparse
import copy
import csv
from datetime import datetime

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

from pyhealth.metrics import binary_metrics_fn

from BIOT.model import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,
    FFCL,
    STTransformer,
    BIOTClassifier,
)
from BIOT.utils import TUABLoader, CHBMITLoader, PTBLoader, BCE


def collate_fn_pad(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])

    X_list, y_list = zip(*batch)

    # X is (channels, time)
    max_len = max(x.shape[-1] for x in X_list)

    X_padded = []
    for x in X_list:
        pad_len = max_len - x.shape[-1]
        if pad_len > 0:
            # Pad last dimension (time) on the right
            x = nn.functional.pad(x, (0, pad_len), value=0)
        X_padded.append(x)

    X_batch = torch.stack(X_padded)

    if isinstance(y_list[0], torch.Tensor):
        y_batch = torch.stack(y_list)
    else:
        y_batch = torch.tensor(y_list)

    return X_batch, y_batch


def evaluate(model, dataloader, device, threshold=None):
    model.eval()
    all_prob = []
    all_gt = []
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            prob = model(X)
            all_prob.append(torch.sigmoid(prob).cpu().numpy())
            all_gt.append(y.cpu().numpy())

    all_prob = np.concatenate(all_prob)
    all_gt = np.concatenate(all_gt)

    if sum(all_gt) * (len(all_gt) - sum(all_gt)) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "pr_auc": 0.0,
            "roc_auc": 0.0,
        }, threshold

    if threshold is None:
        # Determine threshold based on validation set logic
        k = int(np.sum(all_gt))
        if k > 0 and k <= len(all_prob):
            threshold = np.sort(all_prob)[-k]
        else:
            threshold = 0.5

    result = binary_metrics_fn(
        all_gt,
        all_prob,
        metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
        threshold=threshold,
    )
    return result, threshold


def prepare_TUAB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # root = "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed"
    if args.adaptive:
        root = "./datasets/TUH/tuh_eeg_abnormal/v3.0.1/edf/processed3"
    else:
        root = "./datasets/TUH/tuh_eeg_abnormal/v3.0.1/edf/processed"

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    collate_fn = collate_fn_pad if args.adaptive else None

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        collate_fn=collate_fn,
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


def supervised(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get data loaders
    if args.dataset == "TUAB":
        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)

    else:
        raise NotImplementedError

    # define the model
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
        )
        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")

    else:
        raise NotImplementedError
    
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, cooldown=2
    )

    best_val_auroc = -1
    best_threshold = 0.5
    best_model_state = None
    best_val_result = None
    patience = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            X, y = batch
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            prob = model(X)
            loss = BCE(prob, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        val_result, val_threshold = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val AUROC: {val_result['roc_auc']:.4f}")

        # Scheduler
        scheduler.step(val_result['roc_auc'])

        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Record epoch metrics
        epoch_record = {
            'timestamp': run_timestamp,
            'epoch': epoch + 1,
            'train_loss': train_loss,
        }
        for k, v in val_result.items():
            epoch_record[f"val_{k}"] = v

        epoch_log_file = "epoch_log.csv"
        epoch_file_exists = os.path.isfile(epoch_log_file)
        epoch_fieldnames = sorted(epoch_record.keys())

        if epoch_file_exists:
            with open(epoch_log_file, 'r', newline='') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                    if header:
                        epoch_fieldnames = header
                except StopIteration:
                    pass

        with open(epoch_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=epoch_fieldnames, extrasaction='ignore')
            if not epoch_file_exists:
                writer.writeheader()
            writer.writerow(epoch_record)

        # Save checkpoint for each epoch
        torch.save(model.state_dict(), f"checkpoints/{args.model}_{args.dataset}_epoch_{epoch+1}_{run_timestamp}.pt")

        if val_result['roc_auc'] > best_val_auroc:
            best_val_auroc = val_result['roc_auc']
            best_threshold = val_threshold
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_result = val_result
            patience_counter = 0
        elif patience is not None:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
    
    # Test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_result, _ = evaluate(model, test_loader, device, threshold=best_threshold)
    print("Test Result:")
    print(test_result)

    # Record parameters and metrics
    record = vars(args).copy()
    record.update(test_result)
    if best_val_result is not None:
        for k, v in best_val_result.items():
            record[f"val_{k}"] = v
    record['timestamp'] = run_timestamp

    print("-" * 30)
    print("Recording run details:")
    for k, v in record.items():
        print(f"{k}: {v}")
    print("-" * 30)

    log_file = "run_log.csv"
    file_exists = os.path.isfile(log_file)
    fieldnames = sorted(record.keys())

    # If file exists, try to use its header order to avoid CSV corruption
    if file_exists:
        with open(log_file, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                if header:
                    fieldnames = header
            except StopIteration:
                pass

    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
    print(f"Run details saved to {log_file}")


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
    parser.add_argument(
        "--adaptive", action='store_true', default=False
    )
    args = parser.parse_args()
    print(args)

    supervised(args)
