from argparse import ArgumentParser
from pathlib import Path

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
        'CBIOT': {'model': 'CBIOT'},
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
        'ToMeCBIOTr38': {'model': 'ToMeCBIOT', 'k': 38},
        'ToMeCBIOTr19': {'model': 'ToMeCBIOT', 'k': 19},
        'ToMeCBIOTr9': {'model': 'ToMeCBIOT', 'k': 9},
    }

    parser = ArgumentParser()
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--subset', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', default=1e-4)

    parser.add_argument('--dry', action='store_true')

    parsed = parser.parse_args()

    combined_args = {}
    combined_args.update(args)

    parsed_args = dict(vars(parsed))
    parsed_args.pop('dry', None)
    parsed_args.pop('model', None)

    combined_args.update(parsed_args)

    summary_file = Path('.', 'logs', parsed.tag, 'summary.csv')
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    if parsed.model not in model_args:
        raise NotImplementedError("Un-supported model name")
    
    def map_dict_to_args(mapping):
        return [it for k, v in mapping.items() for it in (f'--{k}', f'{v}')]
    
    compiled_args = ( 
        map_dict_to_args(args) 
        + map_dict_to_args(parsed_args) 
        + map_dict_to_args(model_args[parsed.model])
        )
    
    from run_binary_supervised import parse_and_exec
    metric_keys, results = parse_and_exec(compiled_args)

    first_append = not summary_file.exists()
    with open(summary_file, 'a') as handle:
        if first_append:
            handle.write(",".join(['model', 'seed'] + metric_keys))
            handle.write('\n')
            first_append = False

        to_log = [parsed.model, parsed.seed] + [results[it] for it in metric_keys]

        handle.write(",".join([str(it) for it in to_log]))
        handle.write('\n')
        handle.flush()

if __name__ == '__main__':
    main()

