import os
import argparse
from pathlib import Path

import torch
import numpy as np

from torch.utils.data import DataLoader


def set_seeds(device, args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

def prepare_dataloader(args, input_size, mean, std):
    from utils import ImageNetWoofLoader
    train_set, val_set, test_set = ImageNetWoofLoader(args.dataset_dir, input_size, mean, std)

    batch_size=  args.batch_size
    num_workers = args.num_workers
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, test_loader

def prepare_model(args) -> 'torch.nn.Module':
    import torch, tome
    model_name = 'vit_b16_in1k'
    model = torch.hub.load("facebookresearch/swag", model=model_name)
    model.eval()
    # print(len(model.blocks))
    tome.patch.swag(model)
    model.r = args.tome_r
    model.variant = args.tome_variant
    return model

def calculate_metrics(y_hat, y_ground, threshold = None):
    from sklearn.metrics import accuracy_score
    result = {
        "accuracy": accuracy_score(y_ground, y_hat),
    }
    return result, threshold

def run(args):
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}-{args.seed}"
    logdir : str = args.log_dir
    workspace = Path('.', 'workspace', args.workspace)
    log_dir = workspace / 'logs' / logdir / version
    cp_dir = workspace / 'checkpoints' / logdir / version
    prof_dir = workspace / 'profiles' / logdir / version

    # test existance, and fast quit to avoid comflict
    if not args.debug and (log_dir.exists() or cp_dir.exists() or prof_dir.exists()):
        print("workspace {}/{} exists, skipping".format(args.log_dir, version))
        exit()

    print("running {}/{}".format(args.log_dir, version))
    print(args)

    device = torch.device('cuda:0')

    # setup seeds
    set_seeds(device, args)

    # define the model
    model = prepare_model(args)
    model = model.to(device)

    # prepare dataloaders
    train_loader, test_loader, val_loader = prepare_dataloader(
        args, 
        384,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # define optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
    )

    def infer_post_fn(pred, label):
        pred = torch.argmax(pred, -1, True)#.flatten(-2, -1)
        return pred, label
    
    from torch.profiler import profile, ProfilerActivity, record_function, _ExperimentalConfig
    from engine import create_global_context, train_loop, test_loop

    # from torch.utils.tensorboard import SummaryWriter
    ctx = create_global_context(device, log_dir, cp_dir)

    
    # def _train_fn():
    #     if not args.no_train:
    #         train_loop(
    #                 ctx_global=ctx,
    #                 epochs=args.epochs, 
    #                 model = model, 
    #                 optimizer = optimizer,
    #                 scheduler = None,
    #                 train_dataloader = train_loader, 
    #                 val_dataloader = val_loader, 
    #                 loss_fn = loss_fn,
    #                 metric_fn = calculate_metrics,
    #                 metric_comp_fn = compare_metrics,
    #                 infer_post_fn = infer_post_fn,
    #             )
    #     if args.test:
    #         test_loop(
    #             ctx_global=ctx,
    #             model=model,
    #             test_dataloader=test_loader,
    #             infer_post_fn=infer_post_fn,
    #             metric_fn=calculate_metrics,
                
    #         )

    from contextlib import ExitStack
    with ExitStack() as stack:

        if args.profile:
            torch.cuda.memory._record_memory_history()
            perf = stack.enter_context(profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True))

            stack.push(lambda *_: torch.cuda.memory._dump_snapshot(str(prof_dir / "memory.pickle")))
            stack.push(lambda *_: perf.export_chrome_trace(str(prof_dir / 'profile.json')))

        if args.train:
            class FastStop:
                tolerance: int
                # cool_down: int
                count_down: int
                def __init__(self, t):
                    self.tolerance = t
                    # self.cool_down = cd
                    self.count_down = t
                
                def update(self, improved):
                    if improved:
                        self.count_down = self.tolerance
                    else:
                        self.count_down -= 1
                    
                    return self.count_down <= 0

            fast_stop = FastStop(5)

            def compare_metrics(best, current):
                # if best is None:
                #     return True, False

                improved = best is None or current['roc_auc'] > best['roc_auc']

                return improved, fast_stop.update(improved)

            def loss_fn(model, preds, labels):
                loss = BCE(preds, labels)
                for module in model.modules():
                    if hasattr(module, 'compression_loss'):
                        loss = loss + module.compression_loss

                return loss

            train_loop(
                    ctx_global=ctx,
                    epochs=args.epochs, 
                    model = model, 
                    optimizer = optimizer,
                    scheduler = None,
                    train_dataloader = train_loader, 
                    val_dataloader = val_loader, 
                    loss_fn = loss_fn,
                    metric_fn = calculate_metrics,
                    metric_comp_fn = compare_metrics,
                    infer_post_fn = infer_post_fn,
                )

        if args.test:
            test_loop(
                ctx_global=ctx,
                model=model,
                test_dataloader=test_loader,
                infer_post_fn=infer_post_fn,
                metric_fn=calculate_metrics,
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
                        default=4, help="number of workers")
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
    parser.add_argument("--seed", type=int, default=1)
    # parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default='test')
    # parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--workspace", type=str, default='workspace/test')
    parser.add_argument("--dataset_dir", type=str, default="datasets/imagewoof2/")
    parser.add_argument("--subset", type=str, default='')

    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--tome_r", type=int, nargs='+', default=[])
    parser.add_argument("--linear", action='store_true', default=False)
    parser.add_argument("--flash", action='store_true', default=False)

    # parser.add_argument("--load_from_checkpoint", type=str, default=None)
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--profile", action='store_true', default=False)
    # parser.add_argument("--rtl_tome", action='store_true', default=False)
    parser.add_argument("--tome_variant", type=str, nargs='+', default=[])
    parser.add_argument("--cls_token", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    # end of modification

    args = parser.parse_args()

    run(args)
