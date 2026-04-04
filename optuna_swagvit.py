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

def objective(trail : optuna.Trial, args, model, dataset, hooks):
    variant = trail.suggest_categorical('variant', ['kiddp'])
    r = trail.suggest_int(f'{variant}_r', 1, 143)
    # use_cls = trail.suggest_categorical('use_cls', [True, False])
    pivot_factor = trail.suggest_float(f'{variant}_pivot_factor', 0, 1)
    imp_factor = trail.suggest_float(f'{variant}_imp_factor', 0, 1)

    seq = 576
    pivot_nums = []
    imp_nums = []
    rs = []

    left_tokens = seq
    for idx in range(4):
        r_ = min(r, left_tokens // 2)
        rs.append(r_)

        imp_num = min(math.floor(imp_factor * (left_tokens + 1)), left_tokens)
        left_tokens -= r_
        pivot_num = max(math.ceil(pivot_factor * left_tokens), 1)

        trail.set_user_attr(f'{variant}_pn_{idx}', pivot_num)
        trail.set_user_attr(f'{variant}_in_{idx}', imp_num)

        pivot_nums.append(pivot_num)
        imp_nums.append(imp_num)

    # acc = call_model(model, dataset, hooks, variant, 'q', [r], use_cls, pivot_nums, imp_nums)
    rs = [it for r_ in rs for it in [r_, 0, 0]]
    acc = call_model(model, dataset, hooks, rs, f"{variant}[k]", False, pivot_nums, imp_nums)
    return acc / 0.915246, 1 - left_tokens / seq
    
    # return acc / 0.820842, r

def run_optuna():
    from run_infer_swag_imwf import calculate_metrics, prepare_dataloader, prepare_model
    from engine import Hooks
    args = SimpleNamespace(
        model = 'b16_swag',
        dataset = 'ImageNetWoof',
        workspace = 'optuna_hyper_2',
        log_dir = 'b16_swag',
        seed = 0,
        subset = 0,
        batch_size = 128,
        num_workers = 8,
        tome_r = [],
        tome_variant = [],
        dataset_dir = './datasets/imagewoof2/'
    )

    model = prepare_model(args)
    hooks = Hooks(
        calc_metric=partial(calculate_metrics), 
        handle_post_infer_result=lambda pred, label: (pred.argmax(-1, True), label)
    )
    _, dataset, _ = prepare_dataloader(
        args,
        384,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    tpe = optuna.samplers.TPESampler(n_startup_trials=60, multivariate=True, group=True)
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3", 
        directions=['maximize'] * 2, 
        study_name='b16_swag_mo_shared_layer_test',
        load_if_exists=True,
        sampler=tpe
    )

    study.sampler = optuna.samplers.QMCSampler(scramble=True, seed=0)

    study.optimize(partial(objective, args = args, model = model, dataset = dataset, hooks = hooks), n_trials=256)
    study.sampler = tpe
    study.optimize(partial(objective, args = args, model = model, dataset = dataset, hooks = hooks), n_trials=256)


if __name__ == '__main__':
    run_optuna()