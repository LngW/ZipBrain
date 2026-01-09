import os
import argparse
import copy
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np

from thirdparty.BIOT.utils import BCE

from models import load_model_by_args
from datas import prepare_dataloader_by_args

def write_records_to_log(log_file_name, records):
    log_file_name.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file_name, 'a') as f:
        for key, value in records.items():
            f.writelines([key, ':\n'])
            for subk, subv in value:
                f.writelines(['\t', subk, ':\t\t', str(subv), '\n'])
        f.write('\n')
        f.write('-' * 30)
        f.write('\n')


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

def create_optimizer_by_args(args, records, model):
    optim_name = args.optimizer.lower()
    if optim_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError("Optimizers other than Adam and AdamW are not supported now")
    records['optim'] = [
        ('name', args.optimizer),
        ('learning_rate', args.lr),
        ('weight_decay', args.weight_decay),
    ]
    return optimizer

def supervised(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = {}
    records['global'] = [
        ('epoch', args.epochs),
        ('warmup_epochs', args.warmup_epochs),
        ('seed', args.seed)
    ]

    # get data loaders
    train_loader, test_loader, val_loader = prepare_dataloader_by_args(args, records)

    # define the model
    with torch.random.fork_rng([torch.device('cpu')]):
        model = load_model_by_args(args, records)
        model.to(device)

    # define the optimizer
    optimizer = create_optimizer_by_args(args, records, model)

    # define the learning rate scheduler
    tmp = {'mode': 'max', 'factor': 0.5, 'patience': 3, 'cooldown': 3}
    records["sche"] = [('name', 'ReduceLROnPlateau')] + [(k, v) for k, v in tmp.items()]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **tmp
    )

    best_val_auroc = -1
    best_threshold = 0.5
    best_model_state = None
    best_val_result = None
    patience = 10
    patience_counter = 0
    # For logging
    sorted_metrics = None

    # configs for quick stopping
    records['qstp'] = [('patience', patience)]

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = Path(".", "logs", args.tag, f"{run_timestamp}_{args.model}_{args.dataset}.log")
    write_records_to_log( log_file_name, records )

    for epoch in range(args.epochs):
        model.train()
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

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
        if epoch >= args.warmup_epochs:
            scheduler.step(val_result['roc_auc'])

        # Log metrics
        with open(log_file_name, 'a') as f:
            if sorted_metrics is None:
                sorted_metrics = list(val_result.keys())
                sorted_metrics.sort()
                f.write(",\t".join(['epoch', 'timestamp', 'train_loss'] + sorted_metrics))
                f.write("\n")

            line = [epoch + 1, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), train_loss] + [val_result[it] for it in sorted_metrics]
            f.write(',\t'.join([str(it) for it in line]))
            f.write('\n')

        # Save checkpoint for each epoch
        save_dir = Path('./checkpoints/')
        save_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), save_dir / f"{args.tag}_{args.model}_{args.dataset}_{run_timestamp}_epoch_{epoch+1}.pt")

        # Quick-stopping
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
    parser.add_argument("--optimizer", type=str, default='Adam', choices=['Adam', 'AdamW'])
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
    parser.add_argument(
        "--subset", type=str, required=True
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=0, help="number of warmup epochs"
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)

    return supervised(parsed_args)

if __name__ == "__main__":
    parse_and_exec()