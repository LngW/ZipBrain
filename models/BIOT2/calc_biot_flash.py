# /// script
# dependencies = [
#   "pandas", "numpy", "tbparse", "matplotlib", "tqdm"
# ]
# ///

# import numpy as np

from pathlib import Path
import pandas as pd

def calculate_metrics_stats(file_name = None, model_order = None):
    
    df = pd.read_csv(file_name)
    model_col = 'model'
    # metric_cols = ['accuracy','balanced_accuracy','pr_auc','roc_auc', 'sec_per_epoch', 'max_alloc_mem']
    metric_cols = [it for it in df.columns[1:] if it not in ['seed', 'model',]]
    
    print(f"Grouping by: '{model_col}'")
    print(f"Calculating stats for: {metric_cols}")

    if model_order:
        # Convert model column to categorical type with specified order
        df[model_col] = pd.Categorical(df[model_col], categories=model_order, ordered=True)

    stats_df = df.groupby(model_col, observed=True)[metric_cols].agg(['mean', 'std'])

    formatted_df = pd.DataFrame(index=stats_df.index)
    for col in metric_cols:
        if (col, 'mean') in stats_df.columns:
            formatted_df[col] = \
                stats_df[(col, 'mean')].map('{:.4f}'.format) + ' ± ' + stats_df[(col, 'std')].map('{:.4f}'.format)


    formatted_df.to_csv(csv_file.parent / 'model_stats_summary.csv')
    return formatted_df, stats_df

def convert_tb_logs(fetch_dir : Path, dump_file : Path):
    from tbparse import SummaryReader
    from tqdm import tqdm

    cols = 'val_acc,val_bacc,val_pr_auc,val_auroc'.split(',')

    results = {k:[] for k in cols + ['sec_per_epoch', 'mem', 'model', 'seed']}
    for log in (pbar := tqdm([it for sub in fetch_dir.iterdir() for it in sub.iterdir()])):

        exp_name = log.parent.stem
        seed = log.name.split('-')[-1]
        log_name = exp_name + "_" + seed
        pbar.set_postfix({'processing': log_name})

        reader = SummaryReader(log, pivot=False, extra_columns={'wall_time',})
        groups = reader.scalars.groupby('tag')

        group = groups.get_group('val_auroc')
        idx = group['value'].argmax()
        series = group['wall_time']
        sec_per_epoch = (series.max() - series.min()) / (series.count() - 1)
        mem = groups.get_group('Mem/AllocMax')['value'].max()

        results['model'].append(exp_name)
        results['seed'].append(seed)
        results['sec_per_epoch'].append(sec_per_epoch)
        results['mem'].append(mem)

        for it in cols:
            value = groups.get_group(it)['value'].iat[idx]
            results[it].append(value)

    df = pd.DataFrame(results)
    df.to_csv(dump_file)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    base_folder = Path('.', args.folder)
    print("Parsing: {}".format(base_folder))
    fetch_dir = base_folder / 'logs'
    csv_file = base_folder / 'csvs' / 'original_logs.csv'
    csv_file.parent.mkdir(exist_ok=True, parents=True)

    convert_tb_logs(fetch_dir, csv_file)
    calculate_metrics_stats(csv_file)
    # exit()
