import os
import argparse
from pathlib import Path

def set_seeds(args):
    import torch
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    # pass

def prepare_dataloader(args, hooks):
    set_seeds(args)
    # from utils import TUABLoader, CHBMITLoader, PTBLoader
    dataset = args.dataset
    subset = args.subset
    seed = args.seed
    model = args.model

    import sys
    import importlib
    import torch
    from torch.utils.data import Dataset, DataLoader

    if dataset == 'TUAB':
        if model == 'BIOT':
            from thirdparty.BIOT.utils import TUABLoader
            root = './datasets/tuab/biot/' + subset
            root = './datasets/tuab/biot/'

            train_files = os.listdir(os.path.join(root, "train"))
            # np.random.shuffle(train_files)
            # train_files = train_files[:100000]
            val_files = os.listdir(os.path.join(root, "val"))
            test_files = os.listdir(os.path.join(root, "test"))

            print(len(train_files), len(val_files), len(test_files))

            train_set = TUABLoader(os.path.join(root, "train"), train_files, args.sampling_rate)
            test_set = TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate)
            val_set = TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate)
        elif model == 'LaBraM':

            sys.modules['data_processor'] = importlib.import_module('thirdparty.LaBraM.data_processor')

            from thirdparty.LaBraM.utils import prepare_TUAB_dataset, get_input_chans
            # train_dataset, test_dataset, val_dataset = prepare_TUAB_dataset("./datasets/tuab/labrama/")
            train_dataset, test_dataset, val_dataset = prepare_TUAB_dataset("./datasets/tuab/labram/")
            ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                        'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
            ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
            args.nb_classes = 1
            metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]

            input_chs = get_input_chans(ch_names)
            from torch.utils.data import Dataset

            # class Wrapper(Dataset):
            #     def __init__(self, upstream):
            #         self.upstream = upstream

            #     def __getitem__(self, index):
            #         return self.upstream[index], input_chs

            #     def __len__(self):
            #         return len(self.upstream)

            # cds = ConstantDataset()
            train_set = train_dataset
            val_set = val_dataset
            test_set = test_dataset

            from einops import rearrange

            def call_model(model, sample):
                eeg = sample
                eeg = rearrange(eeg.float(), 'B N (A T) -> B N A T', T=200) / 100
                return model(eeg, input_chans = input_chs)

            hooks.call_model = call_model
        elif model == 'EEGPT':
            from thirdparty.EEGPT.downstream_tueg.utils import prepare_TUAB_dataset
            
            train_set, test_set, val_set = prepare_TUAB_dataset('./datasets/tuab/eegpt/')
        elif model == 'TFM':
            sys.modules['utils'] = importlib.import_module('thirdparty.TFM_Tokenizer.utils')
            sys.modules['datasets.data_loaders'] = importlib.import_module('thirdparty.TFM_Tokenizer.datasets.data_loaders')
            sys.modules['models.tfm_token'] = importlib.import_module('thirdparty.TFM_Tokenizer.models.tfm_token')

            from thirdparty.TFM_Tokenizer.datasets.data_loaders import TUABloader
            train_set = TUABloader('./datasets/tuab/tfm_tokenizer/', 'train', 200, None)
            test_set = TUABloader('./datasets/tuab/tfm_tokenizer/', 'test', 200, None)
            val_set = TUABloader('./datasets/tuab/tfm_tokenizer/', 'val', 200, None)

    elif dataset == 'RANDOM':
        g = torch.Generator().manual_seed(seed)
        class RandomDataset(Dataset):
            def __init__(self, n_samples, n_channels, sampling_rate, sample_length):
                self.n_samples = n_samples
                self.data_shape = (n_channels, int(sampling_rate * sample_length))

            def __len__(self):
                return self.n_samples

            def __getitem__(self, index):
                X = torch.randn(self.data_shape, generator=g)
                y = torch.randint(0, 2, (1,), generator=g)#.float()
                return X, y

        train_set = RandomDataset(64, args.in_channels, args.sampling_rate, args.sample_length)
        val_set = RandomDataset(64, args.in_channels, args.sampling_rate, args.sample_length)
        test_set = RandomDataset(64, args.in_channels, args.sampling_rate, args.sample_length)
    else:
        raise NotImplementedError("Unrecognized dataset: {}".format(dataset))

    # prepare training and test data loader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

def prepare_CHB_MIT_dataloader(args):
    import torch
    from torch.utils.data import DataLoader
    import numpy as np
    from thirdparty.BIOT.utils import CHBMITLoader

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
    from thirdparty.BIOT.utils import PTBLoader

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

def prepare_model(args): # -> 'torch.nn.Module':
    import torch
    import patch
    threshold = 0.5
    with torch.random.fork_rng():
        if args.model == "BIOT":
            from thirdparty.BIOT.model import BIOTClassifier
            model = BIOTClassifier(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )

            state_dict = torch.load('models/BIOT2/checkpoints/prest-tuab100.pt', weights_only=True)
            model_dict = state_dict['model']
            threshold = state_dict['threshold']

            if 'biot.window' in model_dict:
                model_dict.pop('biot.window')

            model.load_state_dict(model_dict)
            patch.biot(model)
        elif args.model == 'LaBraM':
            import thirdparty.LaBraM.modeling_finetune
            from timm.models import create_model

            state_dict = torch.load('./finetune/LaBraM/checkpoints/finetune_tuab_base_256/checkpoint-best.pth', weights_only=False)
            args = state_dict['args']
            model_dict = state_dict['model']

            model = create_model(
                args.model,
                pretrained=False,
                num_classes=args.nb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                attn_drop_rate=args.attn_drop_rate,
                drop_block_rate=None,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale,
                use_rel_pos_bias=args.rel_pos_bias,
                use_abs_pos_emb=args.abs_pos_emb,
                init_values=args.layer_scale_init_value,
                qkv_bias=args.qkv_bias,
            )

            model.load_state_dict(model_dict)
            patch.labram(model)

            return model, 0.5

        elif args.model == 'EEGPT':
            from thirdparty.EEGPT.downstream_tueg.Modules.models.EEGPT_mcae_finetune_change import EEGPTClassifier
            use_channels_names = [      
                        'FP1','FPZ', 'FP2',
                'F7', 'F3', 'FZ', 'F4', 'F8',
                'T7', 'C3', 'CZ', 'C4', 'T8',
                'P7', 'P3', 'PZ', 'P4', 'P8',
                        'O1', 'O2' ]
            ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                            'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
            ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
            model = EEGPTClassifier(
                num_classes=1,
                in_channels=len(ch_names), 
                img_size=[len(use_channels_names),2000], 
                use_channels_names=use_channels_names, 
                use_chan_conv=True,
            )

            state_dict = torch.load('./finetune/EEGPT/downstream_tueg/checkpoints/finetune_tuab_eegpt/checkpoint-best.pth', weights_only=False)
            model.load_state_dict(state_dict['model'])

            patch.eegpt(model)

            return model, 0.5

        elif args.model == 'CBraMod':
            pass
        elif args.model == 'TFM':
            from thirdparty.TFM_Tokenizer.tfm_tokenizer_inference import Pl_tfm_tokenizer_inference
            from types import SimpleNamespace
            args = SimpleNamespace()
            args.vqvae_pretrained_path = 'thirdparty/TFM_Tokenizer/pretrained_weigths/single_dataset_settings/TUAB_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth'
            args.code_book_size = 8192
            args.emb_size = 64
            args.finetuned_path = 'thirdparty/TFM_Tokenizer/pretrained_weigths/single_dataset_settings/TUAB_tfm_tokenizer_2x2x8/tfm_encoder_best_model.pth'
            args.resampling_rate = 200

            dataset_params = {
                'classification_task': 'binary',
                'num_classes': 1
            }

            model = Pl_tfm_tokenizer_inference(args, None, 'workspace/debug/', 1, dataset_params)
            patch.tfm_tokenizer(model)

            # return model
        else:
            raise NotImplementedError
        
    return model, threshold

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

def pre_main():
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
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument(
        "--model", type=str, help="which supervised model to use", required=True
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
    parser.add_argument("--pivot_factor", type=float, default=0.05)
    parser.add_argument("--use_cls", type=bool, default=True)
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
    
    main(args)

def main(args):

    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}-{args.seed}"
    version = f"{args.dataset}-{'_'.join(args.tome_variant)}-{'cls' if args.use_cls else 'mean'}-{args.pivot_factor}-{'_'.join([str(it) for it in args.tome_r])}"
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
    from engine import Hooks, create_global_context, destructure_global_context, train, test, valid
    hooks = Hooks(calc_metric=lambda pred, label : calculate_metrics(pred, label, threshold))

    # prepare dataloaders
    train_loader, test_loader, val_loader = prepare_dataloader(args, hooks)

    # define the model
    model, threshold = prepare_model(args)

    # import patch
    # patch.biot(model, True)
    model.r = args.tome_r
    model.variant = args.tome_variant
    model.pivot_factor = args.pivot_factor
    model.use_cls = args.use_cls

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

    # from thirdparty.BIOT.utils import BCE

    def loss_fn(model, preds, labels):
        loss = torch.nn.BCEWithLogitsLoss(preds, labels)
        for module in model.modules():
            if hasattr(module, 'compression_loss'):
                loss = loss + module.compression_loss

        return loss
    
    def compose_custom_state(result, metrics):
        result['threshold'] = metrics['threshold']

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, len(train_loader) * args.epochs)
    scheduler = None


    hooks.calc_loss = loss_fn
    hooks.handle_post_infer_result = infer_post_fn
    hooks.compare_metric = compare_metrics
    hooks.compose_cp_custom_state = compose_custom_state
    hooks.test_break = lambda better, *_: fast_stop.update(better)
    hooks.schedule_step_batch = lambda it : it.step()

    ctx = create_global_context(device, log_dir, cp_dir, 1)

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
    pre_main()
