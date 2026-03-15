import os
import argparse
from pathlib import Path

def set_seeds(args):
    pass

def prepare_dataloader(args):
    # from utils import TUABLoader, CHBMITLoader, PTBLoader
    dataset = args.dataset
    subset = args.subset
    seed = args.seed

    if dataset == 'TUAB':
        return prepare_TUAB_dataloader(args)
    elif dataset == 'RANDOM':
        return prepare_RND_dataloader(args)

    pass

def prepare_model(args) -> 'torch.nn.Module':
    import torch
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
            from model import BIOTClassifier
            model = BIOTClassifier(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
                # linear = args.linear, # modifications for tc_eeg
                # top_k = args.top_k,
                # tome_r = args.tome_r,
                # flash=args.flash,
                # tome_variant=args.tome_variant
            )
            if args.pretrain_model_path and (args.sampling_rate == 200):
                model.biot.load_state_dict(torch.load(args.pretrain_model_path))
                print(f"load pretrain model from {args.pretrain_model_path}")


            # for f, g in model.biot.transformer.layers.layers:
            #     g.fn.fn.dropout.p = 0.5 # PreNorm.Chunk.FeedForward.Dropout.p
            #     pass

        else:
            raise NotImplementedError

    return model

def calculate_metrics(y_hat, y_ground, threshold = None):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score

    if (
        sum(y_ground) * (len(y_ground) - sum(y_ground)) != 0
    ):  # to prevent all 0 or all 1 and raise the AUROC error
        if threshold is None:
            threshold = np.sort(y_hat)[-int(np.sum(y_ground))].item()
        y_pred = np.empty_like(y_hat)
        y_pred[y_hat >= threshold] = 1
        y_pred[y_hat < threshold] = 0
        result = {
            "accuracy": accuracy_score(y_ground, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_ground, y_pred),
            "pr_auc": average_precision_score(y_ground, y_hat),
            "roc_auc": roc_auc_score(y_ground, y_hat),
            "threshold": threshold,
        }
    else:
        result = {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "pr_auc": 0.0,
            "roc_auc": 0.0,
            "threshold": 0.5,
        }
    return result

def prepare_TUAB_dataloader(args):
    import torch
    from torch.utils.data import DataLoader
    # import numpy as np
    from utils import TUABLoader

    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

    # torch.default_generator

    root = "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed"
    root = './datasets/TUAB/processed_' + args.subset

    train_files = os.listdir(os.path.join(root, "train"))
    # np.random.shuffle(train_files)
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
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        # persistent_workers=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        # persistent_workers=True,
        pin_memory=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_CHB_MIT_dataloader(args):
    import torch
    from torch.utils.data import DataLoader
    import numpy as np
    from utils import CHBMITLoader

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
    import torch
    from torch.utils.data import DataLoader
    import numpy as np
    from utils import PTBLoader

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
    import torch
    # from torch.utils.data import DataLoader
    # import numpy as np
    # from utils import CHBMITLoader

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
        val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers
    )

    return train_loader, test_loader, val_loader

def supervised(args):

    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}-{args.seed}"
    logdir : str = args.log_dir
    workspace = Path('.', 'workspace', args.workspace)
    log_dir = workspace / 'logs' / logdir / version
    cp_dir = workspace / 'checkpoints' / logdir / version
    prof_dir = workspace / 'profiles' / logdir / version

    if not args.debug and (log_dir.exists() or cp_dir.exists() or prof_dir.exists()):
        print("workspace {}/{} exists, skipping".format(args.log_dir, version))
        exit()

    print("running {}/{}".format(args.log_dir, version))
    print(args)

    import torch
    # get data loaders
    device = torch.device('cuda:0')

    # prepare dataloaders
    train_loader, test_loader, val_loader = prepare_dataloader(args)

    # define the model
    model = prepare_model(args)

    # load from checkpoint
    state_dict = torch.load('checkpoints/prest-tuab100.pt', weights_only=True)
    model_dict = state_dict['model']
    if 'biot.window' in model_dict:
        model_dict.pop('biot.window')
    model.load_state_dict(model_dict)
    threshold = state_dict['threshold']

    from tome.patch import apply_patch
    apply_patch(model, True)
    model.r = args.tome_r
    model.variant = args.tome_variant

    # define optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
    )

    def infer_post_fn(pred, label):
        pred = torch.sigmoid(pred).flatten()
        return pred, label
    
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
        improved = best is None or current['roc_auc'] > best['roc_auc']

        return improved

    from utils import BCE

    def loss_fn(model, preds, labels):
        loss = BCE(preds, labels)
        for module in model.modules():
            if hasattr(module, 'compression_loss'):
                loss = loss + module.compression_loss

        return loss
    
    def compose_custom_state(result, metrics):
        result['threshold'] = metrics['threshold']

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, len(train_loader) * args.epochs)

    from engine import Hooks, create_global_context, destructure_global_context, train, test, valid

    hooks = Hooks(calc_metric=lambda pred, label : calculate_metrics(pred, label, threshold))
    hooks.calc_loss = loss_fn
    hooks.handle_post_infer_result = infer_post_fn
    hooks.compare_metric = compare_metrics
    hooks.compose_cp_custom_state = compose_custom_state
    hooks.test_break = lambda better, *_: fast_stop.update(better)
    hooks.schedule_step_batch = lambda it : it.step()

    ctx = create_global_context(device, log_dir, cp_dir, 2)

    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.push(lambda *_: destructure_global_context(ctx))
        if args.profile:
            from torch.profiler import profile, ProfilerActivity
            torch.cuda.memory._record_memory_history()
            stack.push(lambda *_: torch.cuda.memory._dump_snapshot(str(prof_dir / 'memory.pickle')))

            prof = None
            stack.push(lambda *_: prof.export_chrome_trace(str(prof_dir / 'profile.json')))
            prof = stack.enter_context(profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]))

        model = model.to(device)
        if args.valid:
            valid(ctx, model, test_loader, hooks)

        if not args.no_train:
            train(ctx, args.epochs, model, train_loader, val_loader, optimizer, hooks, scheduler)

        if args.test:
            test(ctx, model, test_loader, hooks)

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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)

    parser.add_argument("--subset", type=str, default='')

    # parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--tome_r", type=int, nargs='+', default=[])
    parser.add_argument("--tome_variant", type=str, nargs='+', default=[])
    # parser.add_argument("--linear", action='store_true', default=False)
    # parser.add_argument("--flash", action='store_true', default=False)

    # parser.add_argument("--load_from_checkpoint", type=str, default=None)
    parser.add_argument("--no_train", action='store_true', default=False)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--valid", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--profile", action='store_true', default=False)
    # parser.add_argument("--rtl_tome", action='store_true', default=False)
    # parser.add_argument("--cls_token", action='store_true', default=False)
    # end of modification

    args = parser.parse_args()

    if args.debug:
        args.workspace = 'debug'
        args.log_dir = 'debug'
    else:
        required = []
        if args.workspace is None:
            required.append('workspace')
            # parser.error('--workspace is required but found None')
        if args.log_dir is None:
            required.append('log_dir')
            # parser.error('--log_dir is required but found None')
        
        if len(required) > 0:
            parser.error('the following arguments are required: ' + ', '.join(['--' + it for it in required]))

    if args.seed is None:
        if not args.no_train:
            parser.error('the following arguments are required: --seed')
        else:
            args.seed = 0

    supervised(args)
