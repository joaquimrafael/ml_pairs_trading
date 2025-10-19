import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def find_model_dirs(base_dir, dataset_name):
    """Finds all model folders within the specific dataset folder."""
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory {dataset_dir} not found.")
    
    model_dirs = []
    for root, dirs, files in os.walk(dataset_dir):
        if any(f in files for f in ["metrics.csv", "strategies.pkl", "predictions.pkl"]):
            if root not in model_dirs:
                model_dirs.append(root)
                
    return model_dirs

def load_model_data(model_dir):
    """Loads the saved files for a single model."""
    data = {}
    # Metrics CSV
    metrics_path = os.path.join(model_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        data['metrics'] = pd.read_csv(metrics_path)

    # Predictions Pickle
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
    """Loads data from all models found for a specific dataset."""
    model_dirs = find_model_dirs(base_dir, dataset_name)
    if not model_dirs:
        print(f"No model directories found for dataset '{dataset_name}' in '{base_dir}'.")
        return {}
        
    all_models_data = {}
    for dir_path in model_dirs:
        model_name = os.path.basename(dir_path)
        all_models_data[model_name] = load_model_data(dir_path)
        
    return all_models_data

def plot_metrics_comparison(all_models_data, save_path="."):
    """Plots classification metrics (accuracy, precision, recall, f1) comparing models."""
    df_list = []
    for model_name, data in all_models_data.items():
        if 'metrics' in data:
            metrics_df = data['metrics'].copy()
            metrics_df['model'] = model_name
            df_list.append(metrics_df)
            
    if not df_list:
        return
        
    combined_df = pd.concat(df_list, ignore_index=True)
    
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    available_metrics = [m for m in metrics_to_plot if m in combined_df.columns]
    
    if not available_metrics:
        print("Classification metric columns (accuracy, precision, etc.) were not found in 'metrics.csv'.")
        return

    for metric in available_metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="threshold", y=metric, hue="model", data=combined_df)
        plt.title(f"{metric.capitalize()} by Threshold and Model")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Threshold")
        plt.legend(title="Model")
        plt.tight_layout()
        
        filename = os.path.join(save_path, f"{metric}_comparison.png")
        plt.savefig(filename)
        plt.close()
        print(f"Metrics chart saved to: {filename}")


def plot_strategy_profits(all_models_data, save_path="."):
    """Plots total profits and Sharpe Ratios per strategy, comparing models."""
    df_list = []
    for model_name, data in all_models_data.items():
        if 'strategies' in data:
            strategies = data['strategies']
            for strategy_name, s_data in strategies.items():
                df_list.append({
                    "model": model_name,
                    "strategy": strategy_name,
                    "total_profit": sum(s_data.get('total_profit', [0])),
                    "sharpe_ratio": max(s_data.get('sharpe_ratios', [0])) if s_data.get('sharpe_ratios') else 0,
                    "num_trades": sum(s_data.get('num_trade', [0]))
                })
                
    if not df_list:
        print("No 'strategies.pkl' files found to plot.")
        return
        
    df = pd.DataFrame(df_list)
    
    # Total Profit Chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x="strategy", y="total_profit", hue="model", data=df)
    plt.title("Total Profit by Strategy and Model")
    plt.ylabel("Total Profit")
    plt.xlabel("Strategy")
    plt.xticks(rotation=15, ha='right')
    plt.legend(title="Model")
    plt.tight_layout()
    
    profit_filename = os.path.join(save_path, "total_profit_comparison.png")
    plt.savefig(profit_filename)
    plt.close()
    print(f"Profit chart saved to: {profit_filename}")
    
    # Sharpe Ratio Chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x="strategy", y="sharpe_ratio", hue="model", data=df)
    plt.title("Sharpe Ratio by Strategy and Model")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Strategy")
    plt.xticks(rotation=15, ha='right')
    plt.legend(title="Model")
    plt.tight_layout()
    
    sharpe_filename = os.path.join(save_path, "sharpe_ratio_comparison.png")
    plt.savefig(sharpe_filename)
    plt.close()
    print(f"Sharpe Ratio chart saved to: {sharpe_filename}")

def plot_prediction_errors(all_models_data, save_path="."):
    """Plots the distribution of prediction errors (residuals) for each model."""
    plt.figure(figsize=(12, 7))
    has_data = False
    for model_name, data in all_models_data.items():
        if 'predictions' in data and 'true_values' in data['predictions'] and 'predicted_values' in data['predictions']:
            has_data = True
            true_values = np.array(data['predictions']['true_values']).flatten()
            predicted_values = np.array(data['predictions']['predicted_values']).flatten()
            
            if true_values.size != predicted_values.size:
                print(f"Warning: Prediction array sizes for model '{model_name}' do not match. Skipping.")
                continue

            errors = true_values - predicted_values
            sns.kdeplot(errors, label=model_name, fill=True, alpha=0.5)

    if not has_data:
        print("No 'predictions.pkl' files found to plot errors.")
        return

    plt.title("Distribution of Prediction Errors (True - Predicted)")
    plt.xlabel("Error (Residual)")
    plt.ylabel("Density")
    plt.legend(title="Model")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
    
    filename = os.path.join(save_path, "prediction_errors_distribution.png")
    plt.savefig(filename)
    plt.close()
    print(f"Error distribution chart saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Compare metrics and strategy results from different models.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the dataset whose models will be compared (e.g., itau3_4_min).")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory where the dataset folders are located.")
    args = parser.parse_args()
    
    all_models_data = aggregate_models_for_dataset(args.base_dir, args.dataset)
    
    if all_models_data:
        # Create a dedicated directory for saving the plots
        plot_dir = os.path.join(args.base_dir, args.dataset, "comparison_plots")
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Plots will be saved in: {plot_dir}")
        
        plot_metrics_comparison(all_models_data, save_path=plot_dir)
        plot_strategy_profits(all_models_data, save_path=plot_dir)
        plot_prediction_errors(all_models_data, save_path=plot_dir)

if __name__ == "__main__":
    main()