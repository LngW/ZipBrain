import os
import argparse
from pathlib import Path
from run_infer_multiclass import prepare_model, prepare_dataloader, calculate_metrics

import optuna

# def set_seeds(args):
#     import torch
#     seed = args.seed
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # np.random.seed(seed)
#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#     torch.backends.cudnn.benchmark=False
#     torch.backends.cudnn.deterministic=True
#     torch.use_deterministic_algorithms(True)

# def prepare_dataloader(args, hooks):
#     set_seeds(args)

#     dataset = args.dataset
#     model = args.model

#     import sys
#     import importlib
#     import torch
#     from torch.utils.data import Dataset, DataLoader

#     train_set = val_set = test_set = None

#     if dataset == 'TUEV':
#         if model == 'BIOT':
#             from thirdparty.BIOT.utils import TUEVLoader
#             root = './datasets/tuev/biot/'

#             train_files = os.listdir(os.path.join(root, "processed_train"))
#             val_files = os.listdir(os.path.join(root, "processed_eval"))
#             test_files = os.listdir(os.path.join(root, "processed_test"))

#             print(len(train_files), len(val_files), len(test_files))

#             # train_set = TUEVLoader(os.path.join(root, "processed_train"), train_files, args.sampling_rate)
#             test_set = TUEVLoader(os.path.join(root, "processed_test"), test_files, 200)
#             # val_set = TUEVLoader(os.path.join(root, "processed_eval"), val_files, args.sampling_rate)
#         elif model == 'LaBraM':

#             sys.modules['data_processor'] = importlib.import_module('thirdparty.LaBraM.data_processor')

#             from thirdparty.LaBraM.utils import prepare_TUEV_dataset, get_input_chans
#             # train_dataset, test_dataset, val_dataset = prepare_TUAB_dataset("./datasets/tuab/labrama/")
#             train_dataset, test_dataset, val_dataset = prepare_TUEV_dataset("./datasets/tuev/labram/")
#             ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
#                         'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
#             ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
#             args.nb_classes = 6

#             input_chs = get_input_chans(ch_names)

#             # train_set = train_dataset
#             # val_set = val_dataset
#             test_set = test_dataset

#             def call_model(model, sample):
#                 from einops import rearrange
#                 eeg = sample
#                 eeg = rearrange(eeg.float(), 'B N (A T) -> B N A T', T=200) / 100
#                 return model(eeg, input_chans = input_chs)

#             hooks.call_model = call_model
#         # elif model == 'EEGPT':
#         #     from thirdparty.EEGPT.downstream_tueg.utils import prepare_TUAB_dataset
            
#         #     train_set, test_set, val_set = prepare_TUAB_dataset('./datasets/tuab/eegpt/')
#         elif model == 'TFM':
#             sys.modules['utils'] = importlib.import_module('thirdparty.TFM_Tokenizer.utils')
#             sys.modules['datasets.data_loaders'] = importlib.import_module('thirdparty.TFM_Tokenizer.datasets.data_loaders')
#             sys.modules['models.tfm_token'] = importlib.import_module('thirdparty.TFM_Tokenizer.models.tfm_token')

#             from thirdparty.TFM_Tokenizer.datasets.data_loaders import TUEVloader
#             # train_set = None # TUEVloader('./datasets/tuab/tfm_tokenizer/', 'train', 200, None)
#             test_set = TUEVloader('./datasets/tuev/tfm_tokenizer/', 'test', 200, None)
#             # val_set = None # TUEVloader('./datasets/tuab/tfm_tokenizer/', 'val', 200, None)

#     elif dataset == 'RANDOM':
#         seed = args.seed

#         g = torch.Generator().manual_seed(seed)
#         class RandomDataset(Dataset):
#             def __init__(self, n_samples, n_channels, sampling_rate, sample_length):
#                 self.n_samples = n_samples
#                 self.data_shape = (n_channels, int(sampling_rate * sample_length))

#             def __len__(self):
#                 return self.n_samples

#             def __getitem__(self, index):
#                 X = torch.randn(self.data_shape, generator=g)
#                 y = torch.randint(0, 2, (1,), generator=g)
#                 return X, y

#         train_set = RandomDataset(64, args.in_channels, args.sampling_rate, args.sample_length)
#         val_set = RandomDataset(64, args.in_channels, args.sampling_rate, args.sample_length)
#         test_set = RandomDataset(64, args.in_channels, args.sampling_rate, args.sample_length)
#     else:
#         raise NotImplementedError("Unrecognized dataset: {}".format(dataset))

#     # prepare training and test data loader
#     if train_set is not None:
#         train_loader = DataLoader(
#             dataset=train_set,
#             batch_size=args.batch_size,
#             shuffle=True,
#             drop_last=True,
#             num_workers=args.num_workers,
#             persistent_workers=args.num_workers > 0,
#             pin_memory=True,
#         )
#     else:
#         train_loader = None

#     if test_set is not None:
#         test_loader = DataLoader(
#             dataset=test_set,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=args.num_workers,
#             persistent_workers=args.num_workers > 0,
#             pin_memory=True,
#         )
#     else:
#         test_loader = None

#     if val_set is not None:
#         val_loader = DataLoader(
#             dataset=val_set,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=args.num_workers,
#             persistent_workers=args.num_workers > 0,
#             pin_memory=True,
#         )
#     else:
#         val_loader = None

#     def olen(data):
#         return len(data) if data is not None else 0

#     print(olen(train_loader), olen(val_loader), olen(test_loader))
#     return train_loader, test_loader, val_loader

# def prepare_model(args): # -> 'torch.nn.Module':
#     import torch
#     import patch

#     model = args.model
#     dataset = args.dataset

#     with torch.random.fork_rng():
#         if model == "BIOT":

#             if dataset == 'TUAB':
#                 cp_path = 'finetune/BIOT/log/TUAB-BIOT-0.0005-512-200-200-100/checkpoints/epoch=0-step=577.ckpt'
#                 args.n_classes = 1
#             elif dataset == 'TUEV':
#                 cp_path = 'finetune/BIOT/log/TUEV-BIOT-0.0005-512-200-200-100/checkpoints/epoch=2-step=432.ckpt'
#                 args.n_classes = 6
#             else:
#                 raise NotImplementedError("BIOT can only work with TUAB and TUEV in this script now.")

#             args.in_channels = 18
#             args.token_size = 200
#             args.hop_length = 100

#             from thirdparty.BIOT.model import BIOTClassifier
#             model = BIOTClassifier(
#                 n_classes=args.n_classes,
#                 # set the n_channels according to the pretrained model if necessary
#                 n_channels=args.in_channels,
#                 n_fft=args.token_size,
#                 hop_length=args.hop_length,
#             )


#             state_dict = torch.load(cp_path, weights_only=False) # Warning! This may cause Arbitary Code Execution!
#             model_dict = state_dict['state_dict']

#             class wrapper(torch.nn.Module):
#                 def __init__(self, model):
#                     super().__init__()
#                     self.model = model
            
#             wp = wrapper(model)
#             wp.load_state_dict(model_dict)
#             model = wp.model
#             patch.biot(
#                 model, 
#                 trace_source=getattr(args, 'trace_source', False), 
#                 show_shape=getattr(args, 'show_shape', False), 
#                 tome_scheme=getattr(args, 'tome_scheme', False)
#             )

#         elif args.model == 'LaBraM':
#             # This is necessary to register the model to timm, so DO NOT remove it
#             import thirdparty.LaBraM.modeling_finetune
#             from timm.models import create_model

#             if dataset == 'TUAB':
#                 cp_path = './finetune/LaBraM/checkpoints/finetune_tuab_base_256/checkpoint-best.pth'
#             elif dataset == 'TUEV':
#                 cp_path = './finetune/LaBraM/checkpoints/finetune_tuev_base_256/checkpoint-best.pth'
#             else:
#                 raise NotImplementedError()

#             state_dict = torch.load(cp_path, weights_only=False)
#             saved_args = state_dict['args']
#             model_dict = state_dict['model']

#             model = create_model(
#                 saved_args.model,
#                 pretrained=False,
#                 num_classes=saved_args.nb_classes,
#                 drop_rate=saved_args.drop,
#                 drop_path_rate=saved_args.drop_path,
#                 attn_drop_rate=saved_args.attn_drop_rate,
#                 drop_block_rate=None,
#                 use_mean_pooling=saved_args.use_mean_pooling,
#                 init_scale=saved_args.init_scale,
#                 use_rel_pos_bias=saved_args.rel_pos_bias,
#                 use_abs_pos_emb=saved_args.abs_pos_emb,
#                 init_values=saved_args.layer_scale_init_value,
#                 qkv_bias=saved_args.qkv_bias,
#             )

#             model.load_state_dict(model_dict)
#             print(getattr(args, 'show_shape', False))
#             patch.labram(
#                 model, 
#                 trace_source=getattr(args, 'trace_source', False), 
#                 show_shape=getattr(args, 'show_shape', False),
#                 tome_scheme=getattr(args, 'tome_scheme', False)
#             )

#         # elif args.model == 'EEGPT':
#         #     from thirdparty.EEGPT.downstream_tueg.Modules.models.EEGPT_mcae_finetune_change import EEGPTClassifier
#         #     use_channels_names = [      
#         #                 'FP1','FPZ', 'FP2',
#         #         'F7', 'F3', 'FZ', 'F4', 'F8',
#         #         'T7', 'C3', 'CZ', 'C4', 'T8',
#         #         'P7', 'P3', 'PZ', 'P4', 'P8',
#         #                 'O1', 'O2' ]
#         #     ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
#         #                     'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
#         #     ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
#         #     model = EEGPTClassifier(
#         #         num_classes=1,
#         #         in_channels=len(ch_names), 
#         #         img_size=[len(use_channels_names),2000], 
#         #         use_channels_names=use_channels_names, 
#         #         use_chan_conv=True,
#         #     )

#         #     state_dict = torch.load('./finetune/EEGPT/downstream_tueg/checkpoints/finetune_tuab_eegpt/checkpoint-best.pth', weights_only=False)
#         #     model.load_state_dict(state_dict['model'])

#         #     patch.eegpt(model)
#         # elif args.model == 'CBraMod':
#         #     pass
#         elif args.model == 'TFM':
#             from thirdparty.TFM_Tokenizer.tfm_tokenizer_inference import Pl_tfm_tokenizer_inference
#             from types import SimpleNamespace
#             base_path = 'thirdparty/TFM_Tokenizer/pretrained_weigths/single_dataset_settings/TUEV_tfm_tokenizer_2x2x8/'
#             model_args = SimpleNamespace()
#             model_args.vqvae_pretrained_path = base_path + 'tfm_tokenizer_last.pth'
#             model_args.code_book_size = 8192
#             model_args.emb_size = 64
#             model_args.finetuned_path = base_path + 'tfm_encoder_best_model.pth'
#             model_args.resampling_rate = 200

#             dataset_params = {
#                 'classification_task': 'multiclass',
#                 'num_classes': 6
#             }

#             model = Pl_tfm_tokenizer_inference(model_args, None, 'workspace/debug/', 1, dataset_params)
#             patch.tfm_tokenizer(
#                 model, 
#                 trace_source=getattr(args, 'trace_source', False), 
#                 show_shape=getattr(args, 'show_shape', False),
#                 tome_scheme=getattr(args, 'tome_scheme', False)
#             )
#         else:
#             raise NotImplementedError()
        
#     return model

# def calculate_metrics(y_hat, y_ground, threshold = None):
#     from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

#     return {
#         "accuracy": accuracy_score(y_ground, y_hat),
#         "balanced_accuracy": balanced_accuracy_score(y_ground, y_hat),
#         "cohen_kappa": cohen_kappa_score(y_ground, y_hat),
#         "f1_weighted": f1_score(y_ground, y_hat, average='weighted'),
#     }

# def main(args):
#     # version = f"{args.model}-{args.dataset}-{'_'.join(args.tome_variant)}-{'_'.join([str(it) for it in args.tome_r])}-{'cls' if getattr(args, 'use_cls', False) else 'mean'}-{args.pivot_factor}"
#     version = "{}-{}-{}-{}-{}-{}".format(
#         args.model,
#         args.dataset,
#         '_'.join(args.tome_variant),
#         '_'.join([str(it) for it in args.tome_r]),
#         'cls' if getattr(args, 'use_cls', False) else 'mean',
#         getattr(args, 'pivot_factor', 0.)
#     )
#     logdir : str = args.log_dir
#     workspace = Path('.', 'workspace', args.workspace)
#     log_dir = workspace / 'logs' / logdir / version
#     cp_dir = workspace / 'checkpoints' / logdir / version
#     prof_dir = workspace / 'profiles' / logdir / version

#     if not args.debug and (log_dir.exists() or cp_dir.exists() or prof_dir.exists()):
#         print("workspace {}/{} exists, skipping".format(args.log_dir, version))
#         exit()

#     print("running {}/{}".format(args.log_dir, version))
#     print(args)

#     import torch
#     # get data loaders
#     device = torch.device('cuda:0')
#     from engine import Hooks, create_global_context, destructure_global_context, train, test, valid
#     hooks = Hooks(calc_metric=lambda pred, label : calculate_metrics(pred, label))

#     # prepare dataloaders
#     _, test_loader, _ = prepare_dataloader(args, hooks)

#     # define the model
#     model = prepare_model(args)

#     model.r = args.tome_r
#     model.variant = args.tome_variant
#     model.pivot_factor = getattr(args, 'pivot_factor', None)
#     model.use_cls = getattr(args, 'use_cls', False)

#     def infer_post_fn(pred, label):
#         pred = torch.argmax(pred, -1)
#         return pred, label
    
#     hooks.handle_post_infer_result = infer_post_fn
#     ctx = create_global_context(device, log_dir, cp_dir, 1)

#     from contextlib import ExitStack
#     with ExitStack() as stack:
#         stack.push(lambda *_: destructure_global_context(ctx))
#         if args.profile:
#             from torch.profiler import profile, ProfilerActivity
#             torch.cuda.memory._record_memory_history()
#             stack.push(lambda *_: torch.cuda.memory._dump_snapshot(str(prof_dir / 'memory.pickle')))

#             prof = None
#             stack.push(lambda *_: prof.export_chrome_trace(str(prof_dir / 'profile.json')))
#             prof = stack.enter_context(profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]))

#         model = model.to(device)
#         if args.valid:
#             valid(ctx, model, test_loader, hooks)

#         # if not args.no_train:
#         #     train(ctx, args.epochs, model, train_loader, val_loader, optimizer, hooks, scheduler)

#         if args.test:
#             test(ctx, model, test_loader, hooks)

def objective(trial : optuna.Trial, args, model, dataloader, hooks):
    pivot_factor = trial.suggest_float('pivot_factor', 0, 1)
    imp_factor = trial.suggest_float('imp_factor', 0, 1)

    version = "{}-{}-{}-{}-{}-{}-{:1.6f}-{:1.6f}".format(
        args.model,
        args.dataset,
        '_'.join(args.tome_variant) if args.tome_variant else 'baseline',
        '_'.join([str(it) for it in args.tome_r]) if args.tome_r else '0',
        'tome' if getattr(args, 'tome_scheme', False) else 'full',
        'cls' if getattr(args, 'use_cls', False) else 'mean',
        pivot_factor,
        imp_factor
    )
    logdir : str = args.log_dir
    workspace = Path('.', 'workspace', args.workspace)
    log_dir = workspace / 'logs' / logdir / version
    cp_dir = workspace / 'checkpoints' / logdir / version
    prof_dir = workspace / 'profiles' / logdir / version

    model.r = args.tome_r
    model.variant = args.tome_variant
    model.pivot_factor = pivot_factor
    model.imp_factor = imp_factor

    from contextlib import ExitStack
    import torch
    from engine import valid, test, create_global_context, destructure_global_context
    device = torch.device('cuda:0')
    with ExitStack() as stack:
        ctx = create_global_context(device, log_dir, cp_dir, 2)
        stack.push(lambda *_: destructure_global_context(ctx))

        model = model.to(device)
        if args.valid:
            metrics = valid(ctx, model, dataloader, hooks)

        # if not args.no_train:
        #     train(ctx, args.epochs, model, train_loader, val_loader, optimizer, hooks, scheduler)

        if args.test:
            metrics = test(ctx, model, dataloader, hooks)

    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    return metrics['balanced_accuracy'], metrics['cohen_kappa'], metrics['f1_weighted']

def optuna_main(args):
    from engine import Hooks
    from functools import partial
    hooks = Hooks(calc_metric=calculate_metrics, handle_post_infer_result=lambda pred, label: (pred.argmax(-1), label))
    _, dataloader, _ = prepare_dataloader(args, hooks)
    model = prepare_model(args)

    study = optuna.create_study(
        storage=getattr(args, 'storage', 'sqlite:///db.sqlite3'),
        study_name=getattr(args, 'study_name', 'accuracy_{}-{}-{}-{}-{}'.format(args.model, args.dataset, args.tome_variant[0], args.tome_r[0], args.suffix)),
        directions=['maximize'] * 3,
        load_if_exists=True,
        sampler=optuna.samplers.QMCSampler(scramble=True)
    )

    objective_ = partial(objective, args = args, model = model, dataloader = dataloader, hooks = hooks)
    study.optimize(objective_, getattr(args, 'n_trials_qmc', 32))

    study.sampler = optuna.samplers.TPESampler(multivariate=True)
    study.optimize(objective_, getattr(args, 'n_trials_tpe', 32))

def pre_main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--epochs", type=int, default=100,
    #                     help="number of epochs")
    # parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    # parser.add_argument("--weight_decay", type=float,
    #                     default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=4, help="number of workers")
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument(
        "--model", type=str, help="which supervised model to use", required=True
    )
    # parser.add_argument(
    #     "--in_channels", type=int, default=16, help="number of input channels"
    # )
    # parser.add_argument(
    #     "--sample_length", type=float, default=10, help="length (s) of sample"
    # )
    # parser.add_argument(
    #     "--n_classes", type=int, default=1, help="number of output classes"
    # )
    parser.add_argument(
        "--sampling_rate", type=int, default=200, help="sampling rate (r)"
    )
    # parser.add_argument("--token_size", type=int,
    #                     default=200, help="token size (t)")
    # parser.add_argument(
    #     "--hop_length", type=int, default=100, help="token hop length (t - p)"
    # )
    # parser.add_argument(
    #     "--pretrain_model_path", type=str, default="", help="pretrained model path"
    # )

    # modification made for tc_eeg
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)

    # parser.add_argument("--subset", type=str, default='')

    # parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--tome_r", type=int, nargs='+', default=[])
    parser.add_argument("--tome_variant", type=str, nargs='+', default=[])
    # parser.add_argument("--pivot_factor", type=float, default=0.05)
    
    # parser.add_argument("--use_cls", type=bool, default=False)
    # parser.add_argument("--linear", action='store_true', default=False)
    # parser.add_argument("--flash", action='store_true', default=False)

    # parser.add_argument("--load_from_checkpoint", type=str, default=None)
    # parser.add_argument("--no_train", action='store_true', default=False)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--valid", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    # parser.add_argument("--profile", action='store_true', default=False)

    parser.add_argument("--show-shape", action='store_true', default=False, dest='show_shape')
    parser.add_argument("--trace-source", action="store_true", default=False, dest='trace_source')
    parser.add_argument("--tome-scheme", action='store_true', default=False, dest='tome_scheme')
    # parser.add_argument("--rtl_tome", action='store_true', default=False)
    # parser.add_argument("--cls_token", action='store_true', default=False)
    # end of modification

    # sub_adder = parser.add_subparsers(dest='sub_cmd_name', required=False)
    # optuna_parser = sub_adder.add_parser('optuna')
    parser.add_argument("--storage", default = 'sqlite:///db.sqlite3')
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--n_trials_qmc", type=int, default=32)
    parser.add_argument("--n_trials_tpe", type=int, default=32)
    # optuna_parser.add_argument("--model", required=True)
    # optuna_parser.add_argument("--dataset", required=True)
    # optuna_parser.add_argument("--tome_r", nargs='+', required=True)
    # optuna_parser.add_argument("--tome_variant", nargs='+', required=True)

    args = parser.parse_args()
    optuna_main(args)
    # exit()

    # if args.debug:
    #     args.workspace = 'debug'
    #     args.log_dir = 'debug'
    # else:
    #     required = []
    #     if args.workspace is None:
    #         required.append('workspace')
    #         # parser.error('--workspace is required but found None')
    #     if args.log_dir is None:
    #         required.append('log_dir')
    #         # parser.error('--log_dir is required but found None')
        
    #     if len(required) > 0:
    #         parser.error('the following arguments are required: ' + ', '.join(['--' + it for it in required]))

    # if args.seed is None:
    #     if not args.no_train:
    #         parser.error('the following arguments are required: --seed')
    #     else:
    #         args.seed = 0
    
    # main(args)

if __name__ == "__main__":
    pre_main()
