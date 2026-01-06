import os
import argparse
import copy
from datetime import datetime

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn


from thirdparty.BIOT.model import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,
    FFCL,
    STTransformer,
    BIOTClassifier,
)

from model import (
    TopKBiotClassifier, SABiotClassifier
)

from thirdparty.BIOT.utils import TUABLoader, CHBMITLoader, PTBLoader, BCE


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
    from pyhealth.metrics import binary_metrics_fn

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
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # root = "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed"
    # if args.adaptive:
    #     root = "./datasets/TUH/tuh_eeg_abnormal/v3.0.1/edf/processed3"
    # else:
    root = "./datasets/TUH/tuh_eeg_abnormal/v3.0.1/edf/200/processed"

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # collate_fn = None

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        # collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        # collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        # collate_fn=collate_fn,
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

    records = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records['global'] = [
        ('epoch', args.epochs),
        ('seed', args.seed)
    ]

    records['data'] = [
        ('name', args.dataset),
        ('batch_size', args.batch_size), 
        ('num_workers', args.num_workers), 
        ('sampling_rate', args.sampling_rate),
    ]

    # get data loaders
    if args.dataset == "TUAB":
        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)

    else:
        raise NotImplementedError

    records['model'] = [
        ('name', args.model),
        ('in_channels', args.in_channels),
        ('n_classes', args.n_classes),
    ]

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
        records['model'] += [
            ('token_size', args.token_size),
            ('hop_length', args.hop_length),
        ]

        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")
    elif args.model == "TKBIOT":
        model = TopKBiotClassifier(
            n_classes=args.n_classes,
            # set the n_channels according to the pretrained model if necessary
            n_channels=args.in_channels,
            n_fft=args.token_size,
            hop_length=args.hop_length,
            k=args.k
        )
        records['model'] += [
            ('token_size', args.token_size),
            ('hop_length', args.hop_length),
            ('k', args.k)
        ]
    elif args.model == "SABIOT":
        model = SABiotClassifier(
            n_classes=args.n_classes,
            # set the n_channels according to the pretrained model if necessary
            n_channels=args.in_channels,
            n_fft=args.token_size,
            hop_length=args.hop_length,
        )
        records['model'] += [
            ('token_size', args.token_size),
            ('hop_length', args.hop_length),
        ]
    else:
        raise NotImplementedError

    # records['args'].extend([(k, v) for k, v in vars(args).items()])

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    records['optim'] = [
        ('name', 'Adam'),
        ('learning_rate', args.lr),
        ('weight_decay', args.weight_decay)
    ]

    tmp = {'mode': 'max', 'factor': 0.5, 'patience': 3, 'cooldown': 3}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **tmp
    )
    records["sche"] = [('name', 'ReduceLROnPlateau')] + [(k, v) for k, v in tmp.items()]

    best_val_auroc = -1
    best_threshold = 0.5
    best_model_state = None
    best_val_result = None
    patience = 10
    patience_counter = 0

    records['qstp'] = [('patience', patience)]

    from pathlib import Path
    log_file_name = Path(".", "logs", args.tag, f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.model}_{args.dataset}.log")
    log_file_name.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file_name, 'a') as f:
        for key, value in records.items():
            f.writelines([key, ':\n'])
            for subk, subv in value:
                f.writelines(['\t', subk, ':\t\t', str(subv), '\n'])
        f.write('\n')
        f.write('-' * 30)
        f.write('\n')

    sorted_metrics = None

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

        # log & checkpoint
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(log_file_name, 'a') as f:
            if sorted_metrics is None:
                sorted_metrics = list(val_result.keys())
                sorted_metrics.sort()
                f.write(",\t".join(['epoch', 'timestamp', 'train_loss'] + sorted_metrics))
                f.write("\n")

            line = [epoch + 1, run_timestamp, train_loss] + [val_result[it] for it in sorted_metrics]
            f.write(',\t'.join([str(it) for it in line]))
            f.write('\n')

        # Save checkpoint for each epoch
        torch.save(model.state_dict(), f"checkpoints/{args.tag}_{args.model}_{args.dataset}_epoch_{epoch+1}_{run_timestamp}.pt")

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

    # Record best validation and test metrics
    formatted_result = [
        ',\t'.join(['run'] + sorted_metrics),
        ',\t'.join(['val'] + [str(best_val_result[it]) for it in sorted_metrics]),
        ',\t'.join(['test'] + [str(test_result[it]) for it in sorted_metrics])
    ]

    for line in formatted_result:
        print(line)

    with open(log_file_name, 'a') as f:
        f.writelines(["\n", "-" * 30, "\n"])
        for line in formatted_result:
            f.writelines([line, '\n'])

    print(f"Run details saved to {log_file_name}")

    return sorted_metrics, best_val_result

def parse_and_exec(args = None):
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
    parser.add_argument(
        "--tag", type=str, required=True
    )
    parser.add_argument(
        "--k", type=int, default=3
    )
    parser.add_argument(
        "--seed", type=int, required=True
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)

    return supervised(parsed_args)

if __name__ == "__main__":
    parse_and_exec()