import os
import argparse
from pathlib import Path
from run_inference import import_models, prepare_model, prepare_dataloader, calculate_metrics_binary, calculate_metrics_multiclass
import optuna
from functools import partial

def objective(trial: optuna.Trial, args, model, dataloader, hooks):
    # 1. Fixed pivot-factor (requested 0.1)
    pivot_factor = trial.suggest_float('pivot_factor', 0.0, 1.0)

    imp_factor = trial.suggest_float('imp_factor', 0.0, 1.0)

    # Resolve m0-m4
    m_values = []
    m_names = ['m0', 'm1', 'm2', 'm3', 'm4']
    for m in m_names:
        val = trial.suggest_categorical(m, ['x', 'q', 'k', 'v'] + [i0 + i1 for i0 in 'qkv' for i1 in 'hc'])
        m_values.append(val)

    # 3. Format tome_variant as requested: append '[m0,m1,m2,m3,m4]'
    m_str = f"[{','.join(m_values)}]"
    current_tome_variant = [it + m_str + '{bottom_pivot}' for it in args.tome_variant]

    # 4. Preparation from run_optuna_grid_full.py
    version = "{}-{}-{}-{}-{}-{}-{:1.6f}-{:1.6f}".format(
        args.model,
        args.dataset,
        '_'.join(current_tome_variant),
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
    model.variant = current_tome_variant
    model.pivot_factor = pivot_factor
    model.imp_factor = imp_factor
    model.use_cls = getattr(args, 'use_cls', False)

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
        if args.test:
            metrics = test(ctx, model, dataloader, hooks)

    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    # Return objectives as in run_optuna_grid_full.py
    if args.n_classes > 1:
        return metrics['balanced_accuracy'], metrics['cohen_kappa'], metrics['f1_weighted']
    else:
        return metrics['balanced_accuracy'], metrics['pr_auc'], metrics['roc_auc']

def run_staged_optuna(args):
    from engine import Hooks
    hooks = Hooks(None)
    import_models(args)
    _, dataloader, _ = prepare_dataloader(args, hooks)
    model = prepare_model(args)

    if args.n_classes > 1:
        hooks.calc_metric = calculate_metrics_multiclass
        hooks.handle_post_infer_result = lambda pred, label: (pred.argmax(-1), label)
    else:
        hooks.calc_metric = partial(calculate_metrics_binary, threshold = 0.5)
        hooks.handle_post_infer_result = lambda pred, label: (pred.sigmoid().flatten(), label)

    default_name = "{}-{}-{}-{}-{}-{}-staged".format(
        args.model,
        args.dataset,
        args.tome_variant[0] if args.tome_variant else 'baseline',
        args.tome_r[0] if args.tome_variant[0] != 'EEGPT' else args.tome_r[1],
        'cls' if args.use_cls else 'mean',
        'tome' if args.tome_scheme else 'full',
    )
    if args.suffix:
        default_name += f"-{args.suffix}"

    study = optuna.create_study(
        storage=args.storage,
        study_name=getattr(args, 'study_name', None) or default_name,
        directions=['maximize'] * 3,
        load_if_exists=True
    )

    def get_best_params_from_study(study, keys, past = None):
        # For multi-objective, we pick the best based on the first objective (balanced_accuracy)
        if past is not None:
            past = -past
        trials = [t for t in study.trials[past:] if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            return {k: 'x' for k in keys}
        best_trial = max(trials, key=lambda t: t.values[-1])
        return {k: best_trial.params.get(k, 'x') for k in keys}

    # Common parameters for optimize
    # n_trials = args.n_trials_tpe
    obj = partial(objective, args=args, model=model, dataloader=dataloader, hooks=hooks)

    if len(study.trials) <= 135:
        # Stage 1: pivot=0.1, imp=0, tune m0, m1
        print("\n>>> Stage 1: Tuning m0, m1 (imp_factor=0)")
        # stage1 = {'fixed': {'imp_factor': 0.0}, 'tuning': ['m0', 'm1'], 'default_m': 'x'}
        study.sampler = optuna.samplers.GridSampler({
            'pivot_factor': [0.1],
            'imp_factor': [0.0],
            'm0': ['x', 'q', 'k', 'v'] + [i0 + i1 for i0 in 'qkv' for i1 in 'hc'],
            'm1': ['x', 'q', 'k', 'v'] + [i0 + i1 for i0 in 'qkv' for i1 in 'hc'],
            'm2': ['x'],
            'm3': ['x'],
            'm4': ['x'],
        })
        study.optimize(obj, 100)
        best_m = get_best_params_from_study(study, ['m0', 'm1'], 100)

        # Stage 2: pivot=0.1, imp=1, m0/m1 fixed, tune m2
        print(f"\n>>> Stage 2: Tuning m2 (imp_factor=1, m0={best_m['m0']}, m1={best_m['m1']})")
        study.sampler = optuna.samplers.GridSampler({
            'pivot_factor': [0.1],
            'imp_factor': [1.],
            'm0': [best_m['m0']],
            'm1': [best_m['m1']],
            'm2': ['x', 'q', 'k', 'v', 'qh', 'kh', 'vh'],
            'm3': ['x'],
            'm4': ['x'],
        })
        study.optimize(obj, n_trials=7)
        best_m.update(get_best_params_from_study(study, ['m2'], 7))

        # Stage 3: pivot=0.1, imp=0.5, m0-m2 fixed, tune m3, m4
        print(f"\n>>> Stage 3: Tuning m3, m4 (imp_factor=0.5, m0-m2 fixed)")
        for sf in ['', 'h', 'c']:
            study.sampler = optuna.samplers.GridSampler({
                'pivot_factor': [0.1],
                'imp_factor': [0.5],
                'm0': [best_m['m0']],
                'm1': [best_m['m1']],
                'm2': [best_m['m2']],
                'm3': [it + sf for it in 'qkv'],
                'm4': [it + sf for it in 'qkv'],
            })
            study.optimize(obj, n_trials=3)
        study.sampler = optuna.samplers.GridSampler({
            'pivot_factor': [0.1],
            'imp_factor': [0.5],
            'm0': [best_m['m0']],
            'm1': [best_m['m1']],
            'm2': [best_m['m2']],
            'm3': ['x'],
            'm4': ['x'],
        })
        study.optimize(obj, n_trials=1)
        
        best_m.update(get_best_params_from_study(study, ['m3', 'm4'], 28))
        study.enqueue_trial({'pivot_factor': 0.1, 'imp_factor': 1.0, **best_m})
        study.optimize(obj, n_trials=1)
    else:
        trial = [it for it in study.trials if it.state == optuna.trial.TrialState.COMPLETE][-1]
        best_m = { k: trial.params[k] for k in ['m0', 'm1', 'm2', 'm3', 'm4', ] }

    # Stage 4: pivot=0.1, m0-m4 fixed, tune imp_factor
    print(f"\n>>> Stage 4: Tuning imp_factor (m0-m4 fixed)")
    study.sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=25)
    study.sampler = optuna.samplers.PartialFixedSampler(best_m, study.sampler)
    # {
    #     'pivot_factor': [0.1],
    #     'imp_factor': [it / 10.0 for it in range(1, 10) if it != 5],
    #     'm0': [best_m['m0']],
    #     'm1': [best_m['m1']],
    #     'm2': [best_m['m2']],
    #     'm3': [best_m['m3']],
    #     'm4': [best_m['m4']],
    # }
    study.optimize(obj, n_trials=args.n_trials_tpe)

    print("\nStaged Optimization Finished!")

def pre_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sampling_rate", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--tome_r", type=int, nargs='+', default=[])
    parser.add_argument("--tome_variant", type=str, nargs='+', default=[])
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--valid", action='store_true', default=False)
    parser.add_argument("--n_classes", type=int, default=1)
    parser.add_argument("--tome-scheme", action='store_true', default=False, dest='tome_scheme')
    parser.add_argument("--use_cls", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--show-shape", action='store_true', default=False, dest='show_shape')
    parser.add_argument("--trace-source", action="store_true", default=False, dest='trace_source')
    parser.add_argument("--storage", default = 'sqlite:///db.sqlite3')
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--study_name", type=str, default=None)

    parser.add_argument("--n_trials_tpe", type=int, default=32)
    # parser.add_argument("--n_trials_search", type=int, default=32)

    args = parser.parse_args()
    run_staged_optuna(args)

if __name__ == "__main__":
    pre_main()
