from argparse import ArgumentParser
from pathlib import Path

def go_exec(shared, seed, model_arg, summary_file):
    from run_binary_supervised import parse_and_exec
    metric_keys, results = parse_and_exec(shared + ['--seed', str(seed)] + model_arg[1])

    first_append = not summary_file.exists()
    with open(summary_file, 'a') as handle:
        if first_append:
            handle.write(",".join(['model', 'seed'] + metric_keys))
            handle.write('\n')
            first_append = False

        to_log = [model_arg[0], seed] + [results[it] for it in metric_keys]

        handle.write(",".join([str(it) for it in to_log]))
        handle.write('\n')
        handle.flush()

def main():
    args = {
        'dataset': 'TUAB',
        'num_workers': 4,
        'in_channels': 16,
        'sampling_rate': 200,
        'token_size': 200,
        'hop_length': 100,
        'batch_size': 128,
        'warmup_epochs': 0,
        'lr': 1e-4,
    }

    model_args = {
        'BIOT': {'model': 'BIOT'},
        'SABIOT': {'model': 'SABIOT'},
        'T3BIOT': {'model': 'TKBIOT','k': 3},
        'T5BIOT': {'model': 'TKBIOT','k': 5},
        'T7BIOT': {'model': 'TKBIOT','k': 7},
        'T9BIOT': {'model': 'TKBIOT','k': 9},
        'T152BIOT': {'model': 'TKBIOT','k': 152},
        'T76BIOT': {'model': 'TKBIOT','k': 76},
        'T38BIOT': {'model': 'TKBIOT','k': 38},
        'T19BIOT': {'model': 'TKBIOT','k': 19},
        'ToMeBIOTr38': {'model': 'ToMeBIOT','k': 38},
        'ToMeBIOTr19': {'model': 'ToMeBIOT','k': 19},
        'ToMeBIOTr9': {'model': 'ToMeBIOT','k': 9},
    }

    parser = ArgumentParser()
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--subset', type=str, required=True)
    parser.add_argument('--seed', action='extend', type=int, nargs='+', default=[])
    parser.add_argument('--model', action='extend', type=str, nargs='*', default=[])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', default=1e-4)

    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--seed_first', action='store_true')

    parsed = parser.parse_args()

    combined_args = {}
    combined_args.update(args)

    parsed_args = dict(vars(parsed))
    parsed_args.pop('dry', None)
    parsed_args.pop('seed', None)
    parsed_args.pop('seed_first', None)
    parsed_args.pop('model', None)

    combined_args.update(parsed_args)

    summary_file = Path('.', 'logs', parsed.tag, 'summary.csv')
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    seeds_to_run = parsed.seed
    models_to_run = model_args if parsed.model is None else [(it, model_args[it]) for it in parsed.model if it in model_args]
    models_to_run = [(k, [it for subk, subv in v.items() for it in [f'--{subk}', f'{subv}']]) for k, v in models_to_run]

    shared_args = [ it for k, v in combined_args.items() for it in [f'--{k}', f'{v}'] ]
    if parsed.seed_first:
        for seed in seeds_to_run:
            for model_arg in models_to_run:
                go_exec(shared_args, seed, model_arg, summary_file)
    else:
        for model_arg in models_to_run:
            for seed in seeds_to_run:
                go_exec(shared_args, seed, model_arg, summary_file)

if __name__ == '__main__':
    main()

