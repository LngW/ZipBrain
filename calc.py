import pandas as pd
import numpy as np

def calculate_metrics_stats(file_path=None):
    """
    Calculates the mean and standard deviation of metrics for each model.
    
    Args:
        file_path (str): Optional path to a CSV file containing the data.
                         If None, sample data is generated.
    """
    
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    # 2. Identify Columns
    # We assume there is a column identifying the model. 
    # We look for common names or default to the first string column.
    model_col = 'model'
    metric_cols = ['accuracy','balanced_accuracy','pr_auc','roc_auc']
    
    print(f"Grouping by: '{model_col}'")
    print(f"Calculating stats for: {metric_cols}")

    # 3. Calculate Mean and Std
    # We group by the model column and aggregate the metric columns
    stats_df = df.groupby(model_col)[metric_cols].agg(['mean', 'std'])

    # 4. Formatting
    formatted_df = pd.DataFrame(index=stats_df.index)
    for col in metric_cols:
        if (col, 'mean') in stats_df.columns:
            formatted_df[col] = stats_df[(col, 'mean')].map('{:.4f}'.format) + ' ± ' + stats_df[(col, 'std')].map('{:.4f}'.format)

    return formatted_df

if __name__ == "__main__":
    # Run the function (pass a filename like 'results.csv' to use real data)
    results = calculate_metrics_stats('./logs/comp_100/summary.csv')
    
    if results is not None:
        print("\n--- Aggregated Model Metrics ---")
        print(results)
        
        # Optional: Save results to a new CSV
        results.to_csv('model_stats_summary.csv')
