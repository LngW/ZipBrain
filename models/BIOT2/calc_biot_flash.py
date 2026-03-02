# /// script
# dependencies = [
#   "pandas", "numpy", "tbparse", "matplotlib", "tqdm"
# ]
# ///

# import numpy as np
from pathlib import Path

def calculate_metrics_stats(file_name = None, model_order = None):
    import pandas as pd
    
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
    import pandas as pd
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

def draw_plot(csv_file, model_col='model', models_order=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    stats_df = pd.read_csv(csv_file, index_col=0)

    plot_metrics = [m for m in stats_df.columns if m not in ['model', 'seed']]
    print(f"\nGenerating combined plot for: {plot_metrics}")
    for metric in plot_metrics:
        plt.figure(figsize=(36, 12))
        # Load raw data for boxplot
        df = pd.read_csv(csv_file)
        if models_order:
            df[model_col] = pd.Categorical(df[model_col], categories=models_order, ordered=True)

        # Define colors based on suffix
        # def get_color(name):
        #     if str(name).endswith('_flash'): return 'salmon'
        #     if str(name).endswith('_std'): return 'skyblue'
        #     if str(name).endswith('_lin'): return 'lightgreen'
        #     return 'gray'

        def get_color(name):
            if str(name).startswith('baseline'): return 'gray'
            if str(name).startswith('ch_'): return 'skyblue'
            if str(name).startswith('ts_'): return 'lightgreen'
            if str(name).startswith('tome_'): return 'salmon'
            if str(name).startswith('f_'): return 'darkblue'
            if str(name).startswith('l_'): return 'darkgreen'
            if str(name).startswith('lq_'): return 'pink'
            return 'gray'

        # renaming = {'baseline': 'BIOT', 'ch_tome_8421': 'w/ CH ToMe \n (r=8,4,2,1)', 'ch_tome_842': 'w/ CH ToMe \n (r=8,4,2,0)'}
        def rename_model(name):
            return name
            name = str(name).replace('_lin', '')
            if name == 'baseline':
                return 'BIOT'
            elif name.startswith('ch_'):
                rs = list(name.split('_')[-1])
                if len(rs) < 4:
                    rs = rs + ['0'] * (4 - len(rs))
                return 'w/ CH-ToMe\n(r={})'.format(','.join(list(rs)))
            elif name.startswith('ts_'):
                rs = name.split('_')[-1]
                return 'w/ TS-ToMe\n(r={})'.format(','.join(list(rs)))
            elif name.startswith('tome_'):
                rs = name.split('_')[-1]
                return 'w/ ToMe\n(r={})'.format(','.join(list(rs)))
            else:
                return name

        # Create boxplot with custom colors
        bp = df.boxplot(column=metric, by=model_col, patch_artist=True, figsize=(18, 6), return_type='dict')
        
        # Rename x-axis labels
        plt.gca().set_xticklabels([rename_model(label.get_text()) for label in plt.gca().get_xticklabels()])

        # Apply colors to boxes
        for i, box in enumerate(bp[metric]['boxes']):
            model_name = models_order[i] if models_order else df[model_col].unique()[i]
            box.set_facecolor(get_color(model_name))

        plt.title(f'Box Plot of {metric}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(csv_file.parent / f'boxplot_{metric}.png')
        plt.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    base_folder = Path('.', 'workspace', args.folder)
    print("Parsing: {}".format(base_folder))
    fetch_dir = base_folder / 'logs'
    csv_file = base_folder / 'csvs' / 'original_logs.csv'
    csv_file.parent.mkdir(exist_ok=True, parents=True)

    convert_tb_logs(fetch_dir, csv_file)
    calculate_metrics_stats(csv_file)
    # draw_plot(csv_file, models_order=custom_order1)
    # exit()
