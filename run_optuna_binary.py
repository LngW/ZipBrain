import os
import argparse
from pathlib import Path
from run_infer_binary import prepare_dataloader, prepare_model, calculate_metrics

import optuna

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

    return metrics['balanced_accuracy'], metrics['pr_auc'], metrics['roc_auc']

def optuna_main(args):
    from engine import Hooks
    from functools import partial
    hooks = Hooks(calc_metric=lambda pred, label: {}, handle_post_infer_result=lambda pred, label: (pred.sigmoid().flatten(), label))
    _, dataloader, _ = prepare_dataloader(args, hooks)
    model, threshold = prepare_model(args)
    hooks.calc_metric = partial(calculate_metrics, threshold=threshold)

    study_name_base = getattr(args, 'study_name', '{}-{}-{}-{}-{}'.format(args.model, args.dataset, args.tome_variant[0], args.tome_r[0], args.suffix))
    storage = getattr(args, 'storage', 'sqlite:///db.sqlite3')
    study = optuna.create_study(
        storage=storage,
        study_name=study_name_base,
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

    # # if args.seed is None:
    # #     if args.train:
    # #         parser.error('the following arguments are required: --seed')
    # #     else:
    # #         args.seed = 0
    
    # main(args)

if __name__ == "__main__":
    pre_main()
