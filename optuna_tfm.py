import optuna
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
import sys
import importlib
import math

def main(args, r, variant, use_cls, pivot_factor, imp_factor):
    from run_infer_binary import calculate_metrics, prepare_dataloader, prepare_model
    # version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}-{args.seed}"
    version = f"{'_'.join(variant)}-{'cls' if use_cls else 'mean'}-{'_'.join([str(it) for it in pivot_factor])}-{'_'.join([str(it) for it in imp_factor])}-{'_'.join([str(it) for it in r])}"
    logdir : str = 'hyper_optune'
    workspace = Path('.', 'workspace', 'hyper_optune')
    log_dir = workspace / 'logs' / logdir / version
    cp_dir = workspace / 'checkpoints' / logdir / version

    # print("running {}/{}".format(args.log_dir, version))
    # print(args)

    # args = SimpleNamespace(model='TFM', dataset='TUAB', seed=0, subset='')
    import torch
    # get data loaders
    device = torch.device('cuda:0')
    from engine import Hooks, create_global_context, destructure_global_context, test
    # define the model
    model, threshold = prepare_model(args)
    hooks = Hooks(calc_metric=lambda pred, label : calculate_metrics(pred, label, threshold))
    ctx = create_global_context(device, log_dir, cp_dir, 1)

    # prepare dataloaders
    _, test_loader, _ = prepare_dataloader(args, hooks)


    # import patch
    # patch.biot(model, True)
    model.r = r
    model.variant = variant
    model.pivot_factor = pivot_factor
    model.imp_factor = imp_factor
    model.use_cls = use_cls


    with ExitStack() as stack:
        stack.push(lambda *_: destructure_global_context(ctx))
        model = model.to(device)
        metrics = test(ctx, model, test_loader, hooks)

    return metrics['accuracy']

def call_model(variant, metric, r, use_cls, pivot_factor, imp_factor):
    args = SimpleNamespace()
    # args.tome_variant = f"{variant}[{metric}]"
    # args.tome_r = r

    args.model = 'TFM'
    args.dataset = 'TUAB'
    args.workspace = 'optuna_hyper_2'
    args.log_dir = 'TFM'
    args.seed = 0
    args.subset = 0
    args.batch_size = 1
    args.num_workers = 4

    if args.model == 'TFM':
        sys.modules['utils'] = importlib.import_module('thirdparty.TFM_Tokenizer.utils')
        sys.modules['datasets.data_loaders'] = importlib.import_module('thirdparty.TFM_Tokenizer.datasets.data_loaders')
        sys.modules['models.tfm_token'] = importlib.import_module('thirdparty.TFM_Tokenizer.models.tfm_token')

    return main(args, r, f"{variant}[{metric}]", use_cls, pivot_factor, imp_factor)

def objective(trail : optuna.Trial):
    pivot_factor = []
    imp_factor = []
    for idx in range(4):
        pivot_factor.append(trail.suggest_float(f'pivot_factor_{idx}', 0.01, 0.99))
        imp_factor.append(trail.suggest_float(f'imp_factor_{idx}', 0.01, 0.99))

    variant = trail.suggest_categorical('variant', ['kiddl', 'kiddl2'])
    r = trail.suggest_int('r', 19, 75)
    use_cls = trail.suggest_categorical('use_cls', [True, False])

    # idx = 0
    acc = call_model(variant, 'q', [r], use_cls, pivot_factor, imp_factor)
    return acc / 0.820842

if __name__ == '__main__':
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3", 
        direction='maximize', 
        study_name='labram_all_hyper_0',
        load_if_exists=True
        # sampler=optuna.samplers.GridSampler(grids)
    )
    study.optimize(objective, n_trials=200)
