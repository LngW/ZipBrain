import optuna
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
import sys
import importlib
from functools import partial
import math

def call_model(model, dataset, hooks, r, variant, use_cls, pivot_num, imp_num):
    # version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}-{args.seed}"
    version = f"{'_'.join(variant)}-{'cls' if use_cls else 'mean'}-{'_'.join([str(it) for it in pivot_num])}-{'_'.join([str(it) for it in imp_num])}-{r}"
    logdir : str = 'hyper_optune'
    workspace = Path('.', 'workspace', 'hyper_optune')
    log_dir = workspace / 'logs' / logdir / version
    cp_dir = workspace / 'checkpoints' / logdir / version

    import torch
    # get data loaders
    device = torch.device('cuda:0')
    from engine import create_global_context, destructure_global_context, test
    # define the model
    ctx = create_global_context(device, log_dir, cp_dir, 2)

    model.r = r
    model.variant = variant
    model.pivot_num = pivot_num
    model.imp_num = imp_num
    model.use_cls = use_cls

    with ExitStack() as stack:
        stack.push(lambda *_: destructure_global_context(ctx))
        model = model.to(device)
        metrics = test(ctx, model, dataset, hooks)

    return metrics['accuracy']

def objective(trail : optuna.Trial, model, dataset, hooks):
    variant = trail.suggest_categorical('variant', ['kiddp', 'kiddl', 'kiddl2'])
    r = trail.suggest_int(f'{variant}_r', 1, 75)
    # use_cls = trail.suggest_categorical('use_cls', [True, False])
    seq = 304
    pivot_nums = []
    imp_nums = []
    for idx in range(4):
        adj_pivot = 0 if variant == 'kiddl2' else 1

        pivot_factor = trail.suggest_float(f'{variant}_pivot_factor_{idx}', 0, 1)
        imp_factor = trail.suggest_float(f'{variant}_imp_factor_{idx}', 0, 1)

        pivot_num = max(math.ceil(pivot_factor * (seq - (idx + adj_pivot) * r)), 1)
        imp_num = min(math.floor(imp_factor * (seq - idx * r + 1)), seq - idx * r)

        trail.set_user_attr(f'{variant}_pn_{idx}', pivot_num)
        trail.set_user_attr(f'{variant}_in_{idx}', imp_num)

        pivot_nums.append(pivot_num)
        imp_nums.append(imp_num)

    # acc = call_model(model, dataset, hooks, variant, 'q', [r], use_cls, pivot_nums, imp_nums)
    acc = call_model(model, dataset, hooks, r, f"{variant}[q]", False, pivot_nums, imp_nums)
    return acc / 0.820842

def run_optuna():
    from run_infer_binary import calculate_metrics, prepare_dataloader, prepare_model
    from engine import Hooks
    args = SimpleNamespace(
        model = 'TFM',
        dataset = 'TUAB',
        workspace = 'optuna_hyper_2',
        log_dir = 'TFM',
        seed = 0,
        subset = 0,
        batch_size = 256,
        num_workers = 8,
    )

    if args.model == 'TFM':
        sys.modules['utils'] = importlib.import_module('thirdparty.TFM_Tokenizer.utils')
        sys.modules['datasets.data_loaders'] = importlib.import_module('thirdparty.TFM_Tokenizer.datasets.data_loaders')
        sys.modules['models.tfm_token'] = importlib.import_module('thirdparty.TFM_Tokenizer.models.tfm_token')

    model, threshold = prepare_model(args)
    hooks = Hooks(
        calc_metric=partial(calculate_metrics, threshold=threshold), 
        handle_post_infer_result=lambda pred, label: (pred.sigmoid().flatten(), label)
    )
    _, dataset, _ = prepare_dataloader(args, hooks)

    study = optuna.create_study(
        storage="sqlite:///db.sqlite3", 
        direction='maximize', 
        study_name='tfm_all_hyper_12',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=120, multivariate=True, group=True)
    )

    study.optimize(partial(objective, model = model, dataset = dataset, hooks = hooks), n_trials=1200)


if __name__ == '__main__':
    run_optuna()