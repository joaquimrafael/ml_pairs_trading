import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def find_model_dirs(base_dir, dataset_name):
    """Find all model folders inside the specific dataset folder."""
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory {dataset_dir} not found.")
    
    model_dirs = []
    for root, dirs, files in os.walk(dataset_dir):
        if "metrics.csv" in files or "strategies.pkl" in files:
            model_dirs.append(root)
    return model_dirs

def load_model_data(model_dir):
    """Load saved files for a model."""
    data = {}
    # Metrics CSV
    metrics_path = os.path.join(model_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        data['metrics'] = pd.read_csv(metrics_path)
    # Predictions pickle
    predictions_path = os.path.join(model_dir, "predictions.pkl")
    if os.path.exists(predictions_path):
        with open(predictions_path, "rb") as f:
            data['predictions'] = pickle.load(f)
    # Strategies pickle
    strategies_path = os.path.join(model_dir, "strategies.pkl")
    if os.path.exists(strategies_path):
        with open(strategies_path, "rb") as f:
            data['strategies'] = pickle.load(f)
    return data

def aggregate_models_for_dataset(base_dir, dataset_name):
    """Load data from all models found for a specific dataset."""
    model_dirs = find_model_dirs(base_dir, dataset_name)
    all_models_data = {}
    for dir_path in model_dirs:
        model_name = os.path.basename(dir_path)
        all_models_data[model_name] = load_model_data(dir_path)
    return all_models_data

def plot_metrics_comparison(all_models_data):
    """Plot classification metrics (accuracy, precision, recall, f1) comparing models."""
    df_list = []
    for model_name, data in all_models_data.items():
        if 'metrics' in data:
            metrics_df = data['metrics'].copy()
            metrics_df['model'] = model_name
            df_list.append(metrics_df)
    if not df_list:
        print("No metrics.csv files found for this dataset.")
        return
    combined_df = pd.concat(df_list, ignore_index=True)
    
    metrics = ["accuracy", "precision", "recall", "f1"]
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="threshold", y=metric, hue="model", data=combined_df)
        plt.title(f"{metric.capitalize()} by Threshold and Model")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Threshold")
        plt.legend(title="Model")
        plt.tight_layout()
        plt.show()

def plot_strategy_profits(all_models_data):
    """Plot total profits and Sharpe ratios per strategy comparing models."""
    df_list = []
    for model_name, data in all_models_data.items():
        if 'strategies' in data:
            strategies = data['strategies']
            for strategy_name, s_data in strategies.items():
                df_list.append({
                    "model": model_name,
                    "strategy": strategy_name,
                    "total_profit": sum(s_data['total_profit']),
                    "sharpe_ratio": max(s_data['sharpe_ratios']) if s_data['sharpe_ratios'] else 0,
                    "num_trades": sum(s_data['num_trade'])
                })
    if not df_list:
        print("No strategies.pkl files found for this dataset.")
        return
    df = pd.DataFrame(df_list)
    
    plt.figure(figsize=(12,6))
    sns.barplot(x="strategy", y="total_profit", hue="model", data=df)
    plt.title("Total Profit by Strategy and Model")
    plt.ylabel("Total Profit")
    plt.xlabel("Strategy")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12,6))
    sns.barplot(x="strategy", y="sharpe_ratio", hue="model", data=df)
    plt.title("Sharpe Ratio by Strategy and Model")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Strategy")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the dataset whose models will be compared.")
    parser.add_argument("--base_dir", type=str, default="plot",
                        help="Base directory where dataset folders are located.")
    args = parser.parse_args()
    
    all_models_data = aggregate_models_for_dataset(args.base_dir, args.dataset)
    plot_metrics_comparison(all_models_data)
    plot_strategy_profits(all_models_data)

if __name__ == "__main__":
    main()
