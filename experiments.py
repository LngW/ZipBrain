from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':

    pre_defined_seeds = [
        12345856, 85875035, 46812486, 68486431, 86435434, 34525135,
    ]

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

    model_args = [
        ('BIOT', {'model': 'BIOT'}), 
        ('SABIOT', {'model': 'SABIOT'}), 
        ('T3BIOT', {'model': 'TKBIOT','k': 3}),
        ('T5BIOT', {'model': 'TKBIOT','k': 5}),
        ('T7BIOT', {'model': 'TKBIOT','k': 7}),
        ('T9BIOT', {'model': 'TKBIOT','k': 9}),
        ('T152BIOT', {'model': 'TKBIOT','k': 152}),
        ('T76BIOT', {'model': 'TKBIOT','k': 76}),
        ('T38BIOT', {'model': 'TKBIOT','k': 38}),
        ('T19BIOT', {'model': 'TKBIOT','k': 19}),
        ('ToMeBIOTr38', {'model': 'ToMeBIOT','k': 38}),
        ('ToMeBIOTr19', {'model': 'ToMeBIOT','k': 19}),
        ('ToMeBIOTr9', {'model': 'ToMeBIOT','k': 9}),
    ]

    parser = ArgumentParser()
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model', type=str, choices=[it[0] for it in model_args], default=None)
    parser.add_argument('--subset', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--dry', action='store_true')

    parsed = parser.parse_args()

    combined_args = {}
    combined_args.update(args)
    combined_args.update(vars(parsed))

    combined_args.pop('dry', None)
    combined_args.pop('seed', None)

    summary_file = Path('.', 'logs', parsed.tag, 'summary.csv')
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    seeds_to_run = pre_defined_seeds if parsed.seed is None else [parsed.seed]
    models_to_run = model_args if parsed.model is None else [it for it in model_args if it[0] == parsed.model]

    first_append = not summary_file.exists()
    for model_arg in models_to_run:
        run_arg = []
        for k, v in combined_args.items():
            run_arg.append(f'--{k}')
            run_arg.append(f'{v}')

        for k, v in model_arg[1].items():
            run_arg.append(f'--{k}')
            run_arg.append(f'{v}')

        print(model_arg[0], ': ', run_arg)

        if not parsed.dry:
            from run_binary_supervised import parse_and_exec
            for seed in seeds_to_run:
                metric_keys, results = parse_and_exec(run_arg + ['--seed', str(seed)])

                with open(summary_file, 'a') as handle:
                    if first_append:
                        handle.write(",".join(['model', 'seed'] + metric_keys))
                        handle.write('\n')
                        first_append = False

                    to_log = [model_arg[0], seed] + [results[it] for it in metric_keys]

                    handle.write(",".join([str(it) for it in to_log]))
                    handle.write('\n')
                    handle.flush()

