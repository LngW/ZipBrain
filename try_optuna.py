import optuna
from pathlib import Path

def main(args):
    from run_infer_binary import calculate_metrics, prepare_dataloader, prepare_model
    # version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}-{args.seed}"
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
    # scheduler = None


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
        model = model.to(device)
        metrics = test(ctx, model, test_loader, hooks)

    return metrics['accuracy']

def call_model(pivot_factor, use_cls, variant, metric):
    from types import SimpleNamespace
    args = SimpleNamespace()
    args.pivot_factor = pivot_factor
    args.use_cls = use_cls
    args.tome_variant = f"{variant}[{metric}]"

    args.model = 'LaBraM'
    args.dataset = 'TUAB'
    args.workspace = 'optuna_hyper'
    args.test = True
    args.debug = False
    args.log_dir = 'LaBraM'
    args.seed = 0
    args.subset = 0

    num = []
    for r in [[3], [7], [11], [15], [19] * 11 + [10]]:
        args.tome_r = r
        num.append(main(args))

    return sum(num) / len(num)

def objective(trail : optuna.Trial):
    pivot_factor = trail.suggest_float('pivot_factor', 0.01, 1)
    use_cls = trail.suggest_categorical('use_cls', [True, False])
    variant = trail.suggest_categorical('variant', ['kiddp', 'kiddl'])
    metric = trail.suggest_categorical('metric', ['q', 'k', 'v', 'x'])

    return call_model(pivot_factor, use_cls, variant, metric)


if __name__ == '__main__':
    # study = optuna.create_study(storage="sqlite:///db.sqlite3", direction='maximize')
    # study.optimize(objective, n_trials=100)
    call_model(0.05, True, 'kiddp', 'q')