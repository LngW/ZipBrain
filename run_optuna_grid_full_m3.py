import os
import argparse
from pathlib import Path
from run_inference import import_models, prepare_model, prepare_dataloader, calculate_metrics_binary, calculate_metrics_multiclass

import optuna

from run_optuna_copy_study import copy_stage1_stage2, copy_stage2_stage3, copy_stage3_stage4

m0_m1 = [
    ('x', 'x')
] + [
    (it + suffix, it + suffix) for it in 'qkv' for suffix in ['', 'h']
] + [
    (it + suffix, 'x') for it in 'qkv' for suffix in ['', 'h']
] + [
    ('x', it + suffix) for it in 'qkv' for suffix in ['', 'h']
] + [
    (it + suffix, 'v' + suffix) for it in 'qk' for suffix in ['', 'h']
] + [
    ('v' + suffix, it + suffix) for it in 'qk' for suffix in ['', 'h']
]

m0_m1_pf = [','.join([it[0], it[1], str(pf)]) for it in m0_m1 for pf in range(0, 11)]

m2_list = [it + suffix for it in 'qkv' for suffix in ['', 'h']] + ['x']

m3_m4 = [
    ('x', 'x')
] + [
    ('q' + suffix, 'k' + suffix) for suffix in ['', 'h']
]

m3_m4_if = [
    ','.join([it[0], it[1], str(pi)]) for it in m3_m4 for pi in range(6)
]

def evaluate_model(args, model, dataloader, hooks, m0, m1, m2, m3, m4, pivot_factor, imp_factor):
    variant = args.tome_variant[0]
    variant = "{}[{}]".format(variant, ",".join([m0, m1, m2, m3, m4]))

    version = "{}-{}-{}-{}-{}-{}-{:1.6f}-{:1.6f}".format(
        args.model,
        args.dataset,
        variant,
        '_'.join([str(it) for it in args.tome_r]) if args.tome_r else '0',
        'tome' if getattr(args, 'tome_scheme', False) else 'full',
        'cls' if getattr(args, 'use_cls', False) else 'mean',
        pivot_factor,
        imp_factor
    )

    logdir : str = args.log_dir
    if args.workspace is not None:
        workspace = Path('.', 'workspace', args.workspace)
        log_dir = workspace / 'logs' / logdir / version if logdir else None
        cp_dir = workspace / 'checkpoints' / logdir / version if logdir else None
        prof_dir = workspace / 'profiles' / logdir / version if logdir else None
    else:
        log_dir = None
        cp_dir = None

    model.r = args.tome_r
    model.variant = variant
    model.pivot_factor = pivot_factor
    model.imp_factor = imp_factor
    model.use_cls = getattr(args, 'use_cls', False)

    if getattr(args, 'fml1', False):
        model.imp_factor = [1] + [imp_factor] * 24
    # model._pinfo['fml1'] = getattr(args, 'fml1', False)

    from contextlib import ExitStack
    import torch
    from engine import valid, test, create_global_context, destructure_global_context
    device = torch.device('cuda:0')
    metrics = {}
    with ExitStack() as stack:
        ctx = create_global_context(device, log_dir, cp_dir, 2)
        stack.push(lambda *_: destructure_global_context(ctx))

        model = model.to(device)
        if args.valid:
            metrics = valid(ctx, model, dataloader, hooks)

        if args.test:
            metrics = test(ctx, model, dataloader, hooks)

    return metrics

def objective0(trial : optuna.Trial, args, model, dataloader, hooks, sub_m0_m1 = None):

    pivot_factor = trial.suggest_float('pivot_factor', 0, 1, step=0.1)
    imp_factor   = trial.suggest_float('imp_factor', 0.6, 1, step=0.1)

    if sub_m0_m1 is None:
        sub_m0_m1 = [','.join(it) for it in m0_m1]
    print(sub_m0_m1)
    m0, m1 = trial.suggest_categorical('m0_m1', sub_m0_m1).split(',')
    m2 = trial.suggest_categorical('m2', m2_list)
    m3, m4 = trial.suggest_categorical('m3_m4', [','.join(it) for it in m3_m4]).split(',')

    metrics = evaluate_model(args, model, dataloader, hooks, m0, m1, m2, m3, m4, pivot_factor, imp_factor)
    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    if args.n_classes > 1:
        return metrics['balanced_accuracy'], metrics['cohen_kappa'], metrics['f1_weighted']
    else:
        return metrics['balanced_accuracy'], metrics['pr_auc'], metrics['roc_auc']

def objective1(trial : optuna.Trial, args, model, dataloader, hooks, sub_m0_m1_fp : list[str]):

    # pivot_factor = trial.suggest_float('pivot_factor', 0, 1, step=0.1)
    imp_factor   = trial.suggest_float('imp_factor', 0.6, 1, step=0.1)

    # if sub_m0_m1 is None:
    #     sub_m0_m1 = [','.join(it) for it in m0_m1]
    # print(sub_m0_m1)
    # m0, m1 = trial.suggest_categorical('m0_m1', sub_m0_m1).split(',')
    m0, m1, fp = trial.suggest_categorical('m0_m1_fp', sub_m0_m1_fp).split(',')
    m2 = trial.suggest_categorical('m2', m2_list)
    m3, m4 = trial.suggest_categorical('m3_m4', [','.join(it) for it in m3_m4]).split(',')

    metrics = evaluate_model(args, model, dataloader, hooks, m0, m1, m2, m3, m4, int(fp) / 10, imp_factor)
    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    if args.n_classes > 1:
        return metrics['balanced_accuracy'], metrics['cohen_kappa'], metrics['f1_weighted']
    else:
        return metrics['balanced_accuracy'], metrics['pr_auc'], metrics['roc_auc']

def objective2(trial : optuna.Trial, args, model, dataloader, hooks, sub_m0_m1_fp_m2 : list[str]):

    m0, m1, fp, m2 = trial.suggest_categorical('m0_m1_fp_m2', sub_m0_m1_fp_m2).split(',')

    m3, m4 = trial.suggest_categorical('m3_m4', [','.join(it) for it in m3_m4]).split(',')
    imp_factor   = trial.suggest_float('imp_factor', 0, 1, step=0.1)

    metrics = evaluate_model(args, model, dataloader, hooks, m0, m1, m2, m3, m4, int(fp) / 10, imp_factor)
    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    if args.n_classes > 1:
        return metrics['balanced_accuracy'], metrics['cohen_kappa'], metrics['f1_weighted']
    else:
        return metrics['balanced_accuracy'], metrics['pr_auc'], metrics['roc_auc']

def objective3(trial : optuna.Trial, args, model, dataloader, hooks, sub_ms):
    m0, m1, m2, m3, m4 = trial.suggest_categorical('ms', sub_ms).split(',')
    pivot_factor = trial.suggest_float('pivot_factor', 0, 1, step=0.1)
    imp_factor   = trial.suggest_float('imp_factor', 0, 1, step=0.1)

    metrics = evaluate_model(args, model, dataloader, hooks, m0, m1, m2, m3, m4, pivot_factor, imp_factor)
    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    if args.n_classes > 1:
        return metrics['balanced_accuracy'], metrics['cohen_kappa'], metrics['f1_weighted']
    else:
        return metrics['balanced_accuracy'], metrics['pr_auc'], metrics['roc_auc']

def optuna_main(args):
    from engine import Hooks
    from functools import partial
    hooks = Hooks(lambda *_: {})
    import_models(args)
    _, dataloader, _ = prepare_dataloader(args, hooks)
    model = prepare_model(args)

    if args.n_classes > 1:
        hooks.calc_metric = calculate_metrics_multiclass
        hooks.handle_post_infer_result = lambda pred, label: (pred.argmax(-1), label)
    else:
        hooks.calc_metric = partial(calculate_metrics_binary, threshold = 0.5)
        hooks.handle_post_infer_result = lambda pred, label: (pred.sigmoid().flatten(), label)

    default_name = "{}-{}-{}-{}-{}".format(
        args.model,
        args.dataset,
        args.tome_variant[0],
        args.tome_r[1] if args.model == 'EEGPT' else args.tome_r[0],
        'tome' if args.tome_scheme else 'full',
    )

    if args.suffix is not None:
        default_name = default_name + '-' + str(args.suffix)

    study_name = getattr(args, 'study_name', default_name)

    if hasattr(args, 'namespace'):
        study_name = '{}:{}'.format(args.namespace, study_name)


    storage = getattr(args, 'storage', 'sqlite:///db.sqlite3')


    def create_study(study_name):
        study_names = optuna.get_all_study_names(storage = storage)
        if study_name in study_names:
            study = optuna.load_study(
                storage=storage,
                study_name=study_name,
                sampler = None
            )
        else:
            study = optuna.create_study(
                storage=storage,
                study_name=study_name,
                directions=['maximize'] * 3,
                sampler=None
            )
        return study

    name_s1, name_s2, name_s3, name_s4 = [('{}-stage{}'.format(study_name, it)) for it in '1234']

    # stage 1: grid search for m0 and m1
    if not getattr(args, 'skip_s1', False):
        study1 = create_study(name_s1)
        sampler_grid = optuna.samplers.GridSampler({
            'm0_m1': [','.join(it) for it in m0_m1],
            'pivot_factor': [0.1 * it for it in range(11)],
            'm2': ['x'],
            'm3_m4': ['x,x'],
            'imp_factor': [1],
        })
        study1.sampler = sampler_grid
        objective_ = partial(objective0, args = args, model = model, dataloader = dataloader, hooks = hooks)
        study1.optimize(objective_, len(m0_m1) * 11 + 1)

    # stage 2: grid search for m2
    # n_trials = getattr(args, 'skip_s2', 0)
    if not getattr(args, 'skip_s2', False):
        need_copy = name_s2 not in optuna.get_all_study_names(storage)
        can_copy = name_s1 in optuna.get_all_study_names(storage)

        if need_copy and not can_copy:
            raise Exception('Stage 1 study does not exists: {}'.format(study_name))
        elif need_copy:
            copy_stage1_stage2(storage, name_s1, name_s2)

        # The stage will be an tpe search, with m0 and m1 fixed
        study2 = create_study(name_s2)
        m0_m1_fp_list = study2.user_attrs['m0_m1_fp'].split('/')
        # num_of_copied = study2.user_attrs['num_of_copied']

        # sampler_tpe = optuna.samplers.TPESampler(
        #     multivariate=True, 
        #     group=True, 
        #     n_startup_trials=num_of_copied + int(n_trials / 3)
        # )

        # sampler_fixed = optuna.samplers.PartialFixedSampler({
        #     # 'm0_m1': m0_m1_list,
        #     'm3_m4': 'x,x',
        #     'imp_factor': 1,
        # }, sampler_tpe)

        sampler_grid = optuna.samplers.GridSampler({
            'm0_m1_fp': m0_m1_fp_list,
            'm2': m2_list,
            'm3_m4': ['x,x'],
            'imp_factor': [1.0],
        })

        study2.sampler = sampler_grid
        objective_ = partial(objective1, args = args, model = model, dataloader = dataloader, hooks = hooks, sub_m0_m1_fp = m0_m1_fp_list)
        study2.optimize(objective_, n_trials=len(m0_m1_fp_list) * len(m2_list) + 1)

    # stage 3: grid for m3, m4, and imp_factor
    if not getattr(args, 'skip_s3', False):
        need_copy = name_s3 not in optuna.get_all_study_names(storage)
        can_copy = name_s2 in optuna.get_all_study_names(storage)

        if need_copy and not can_copy:
            raise Exception('Stage 2 study does not exists: {}'.format(name_s2))
        elif need_copy:
            copy_stage2_stage3(storage, name_s2, name_s3)

        study3 = create_study(name_s3)
        m0_m1_fp_m2_list : list[str] = study3.user_attrs['m0_m1_fp_m2'].split('/')
        
        sampler_grid = optuna.samplers.GridSampler({
            'm0_m1_fp_m2': m0_m1_fp_m2_list,
            'm3_m4': [','.join(it) for it in m3_m4],
            'imp_factor': [it / 10 for it in range(11)]
        })

        study3.sampler = sampler_grid
        objective_ = partial(objective2, args = args, model = model, dataloader = dataloader, hooks = hooks, sub_m0_m1_fp_m2 = m0_m1_fp_m2_list)
        study3.optimize(objective_, n_trials=len(m0_m1_fp_m2_list) * len(m3_m4) * 11 + 1)
    
    if not getattr(args, 'skip_s4', False):
        need_copy = name_s4 not in optuna.get_all_study_names(storage)
        can_copy = name_s3 in optuna.get_all_study_names(storage)

        if need_copy and not can_copy:
            raise Exception('Stage 4 study does not exist: {}'.format(name_s4))
        elif need_copy:
            copy_stage3_stage4(storage, name_s3, name_s4)

        study4 = create_study(name_s4)
        ms_list = study4.user_attrs['ms'].split('/')

        sampler_grid = optuna.samplers.GridSampler({
            'ms': ms_list,
            'pivot_factor': [it / 10 for it in range(11)],
            'imp_factor': [it / 10 for it in range(11)]
        })

        study4.sampler = sampler_grid
        objective_ = partial(objective3, args = args, model = model, dataloader = dataloader, hooks = hooks, sub_ms = ms_list)
        study4.optimize(objective_, n_trials=len(ms_list) * 11 * 11 + 1)

    # stage 3: tpe for m3, m4 and imp_factor
    # n_trials = getattr(args, 'n_trials_tpe_s2', 0)
    # if n_trials > 0:
    #     names = optuna.get_all_study_names(storage)
    #     need_copy = (study_name + '-stage2') not in names
    #     can_copy = (study_name + '-stage1') in names
    #     if need_copy and not can_copy:
    #         raise Exception('Stage 1 study does not exists: {}'.format(study_name))
    #     elif need_copy:
    #         copy_stage2_stage3(storage, study_name + '-stage1', study_name + '-stage2')
        
    
    # stage 2
    # if n_trials > 0:
    #     need_copy = (study_name + '-stage2') not in optuna.get_all_study_names(storage)
    #     can_copy = (study_name + '-stage1') in optuna.get_all_study_names(storage)

    #     study2 = create_study(study_name + '-stage2')
    #     if need_copy:
    #         if not can_copy:
    #             raise Exception('Stage 1 study does not exists: {}'.format(study_name))
            
    #         study1 = create_study(study_name + '-stage1')
    #         trials = list(filter(lambda it: it.state == optuna.trial.TrialState.COMPLETE, study1.trials))[:297]
    #         best_trial_v2 = sorted(trials, key=lambda key: key.values[-1], reverse=True)
    #         best_trial_v1 = sorted(trials, key=lambda key: key.values[-2], reverse=True)

    #         sub_m0_m1 = {it.params['m0_m1'] for it in best_trial_v1[:5]} | {it.params['m0_m1'] for it in best_trial_v2[:5]}
    #         sub_m0_m1_list = list(sub_m0_m1)

    #         def convert_trials(trial : optuna.trial.Trial):
    #             distributions = dict(trial.distributions)
    #             distributions['m0_m1'] = optuna.distributions.CategoricalDistribution(sub_m0_m1_list)

    #             return optuna.trial.create_trial(
    #                 values = trial.values,
    #                 params = trial.params,
    #                 user_attrs = trial.user_attrs,
    #                 system_attrs = trial.system_attrs,
    #                 distributions = distributions
    #             )

    #         sub_trials = [convert_trials(t) for t in filter(lambda it: it.params['m0_m1'] in sub_m0_m1, trials)]
    #         study2.add_trials(sub_trials)
    #         study2.set_user_attr('m0_m1_sublist', '/'.join(sub_m0_m1_list))
    #         study2.set_user_attr('num_of_copied', len(sub_trials))
        
    #     sub_m0_m1_list = study2.user_attrs['m0_m1_sublist'].split('/')
    #     sampler_tpe = optuna.samplers.TPESampler(multivariate=True, n_startup_trials = int(n_trials / 3) + study2.user_attrs['num_of_copied'], seed=getattr(args, 'seed', 42))
    #     study2.sampler = sampler_tpe
    #     objective_ = partial(objective, args = args, model = model, dataloader = dataloader, hooks = hooks, sub_m0_m1 = sub_m0_m1_list)
    #     study2.optimize(objective_, n_trials)

    # # stage 2
    # study2 = create_study(study_name + '-stage2')

    # if len(study2.trials) == 0:
    #     def convert_trials(trial : optuna.trial.Trial):
    #         distributions = dict(trial.distributions)
    #         distributions['m0_m1'] = optuna.distributions.CategoricalDistribution(sub_m0_m1_list)

    #         return optuna.trial.create_trial(
    #             values = trial.values,
    #             params = trial.params,
    #             user_attrs = trial.user_attrs,
    #             system_attrs = trial.system_attrs,
    #             distributions = distributions
    #         )

    #     study2.add_trials([convert_trials(t) for t in filter(lambda it: it.params['m0_m1'] in sub_m0_m1, trials)])

    # sampler_tpe = optuna.samplers.TPESampler(multivariate=True, n_startup_trials = int(n_trials / 3), seed=getattr(args, 'seed', 42))
    # study2.sampler = sampler_tpe
    # objective_ = partial(objective, args = args, model = model, dataloader = dataloader, hooks = hooks, sub_m0_m1 = sub_m0_m1_list)
    # study2.optimize(objective_, n_trials)

    # n_trials = getattr(args, 'n_trials_tpe_s2', 32)
    # sampler_tpe = optuna.samplers.TPESampler(multivariate=True, n_startup_trials = int(n_trials / 3), seed=getattr(args, 'seed', 42))
    # best_trial = max(study.best_trials, key=lambda key: key.values[-1])
    # study.sampler = optuna.samplers.PartialFixedSampler({
    #     'm0_m1': best_trial.params["m0_m1"],
    #     'm2': best_trial.params["m2"],
    #     'pivot_factor': best_trial.params["pivot_factor"],
    # }, sampler_tpe)
    # study.optimize(objective_, n_trials)

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
    parser.add_argument("--fml1", action='store_true', default=False)
    
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
    parser.add_argument("--namespace", default = 'default')
    parser.add_argument("--suffix", type=str, default=None)
    # parser.add_argument("--n_trials_qmc", type=int, default=32)
    parser.add_argument("--skip-s1", action='store_true', default=False, dest='skip_s1')
    parser.add_argument("--skip-s2", action='store_true', default=False, dest='skip_s2')
    parser.add_argument("--skip-s3", action='store_true', default=False, dest='skip_s3')
    parser.add_argument("--skip-s4", action='store_true', default=False, dest='skip_s4')
    # parser.add_argument("--n_trials_tpe_s3", type=int, default=0)
    # parser.add_argument("--n_trials_tpe_s2", type=int, default=32)
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
