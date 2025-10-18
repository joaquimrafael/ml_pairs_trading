import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# General Seaborn style configuration
sns.set_theme(style="whitegrid", palette="crest")


data_set = ""

def set_data_set(data_set_name):
    global data_set
    data_set = data_set_name

#Plots for the SL

def ensure_model_dir(model_name):
    """
    Ensures that the directory 'plot/<dataset_name>/<model_name>/' exists.
    Returns the path to the model's plot directory.
    """
    dataset_name = os.path.splitext(os.path.basename(data_set))[0]

    model_dir = os.path.join("plot", dataset_name, model_name)
    os.makedirs(model_dir, exist_ok=True)

    return model_dir


def plot_model_metrics(metrics_dict, model_name, title="Evaluation Metrics"):
    """
    Plots a bar chart with the model's error metrics and saves it in the model's folder.
    """
    save_path = os.path.join(ensure_model_dir(model_name), "metrics_plot.png")

    plt.figure(figsize=(8, 5))
    plt.bar(metrics_dict.keys(), metrics_dict.values(), color='skyblue')
    plt.title(title)
    plt.ylabel('Error')
    for i, v in enumerate(metrics_dict.values()):
        plt.text(i, v + max(metrics_dict.values()) * 0.01, f"{v:.4f}", ha='center')

    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def plot_predictions(true_values, predicted_values, model_name):
    """
    Plots the time series of true vs predicted values and saves it in the model's folder.
    """
    save_path = os.path.join(ensure_model_dir(model_name), "predictions_vs_true.png")

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=np.arange(len(true_values)), y=true_values, label="True", color="blue")
    sns.lineplot(x=np.arange(len(predicted_values)), y=predicted_values, label="Predicted", color="red")
    plt.title("True vs Predicted Values")
    plt.xlabel("Observations")
    plt.ylabel("Ratio")
    plt.legend()

    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def plot_error_distribution(true_values, predicted_values, model_name):
    """
    Shows the distribution of prediction errors (difference between true and predicted values)
    and saves it in the model's folder.
    """
    save_path = os.path.join(ensure_model_dir(model_name), "error_distribution.png")

    errors = np.array(predicted_values) - np.array(true_values)
    plt.figure(figsize=(8, 5))
    sns.histplot(errors, kde=True, bins=30, color="orange")
    plt.title("Prediction Error Distribution")
    plt.xlabel("Error (Predicted - True)")
    plt.ylabel("Frequency")

    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def plot_correlation_heatmap(true_values, predicted_values, model_name):
    """
    Shows a heatmap of the correlation between true and predicted values and saves it in the model's folder.
    """
    save_path = os.path.join(ensure_model_dir(model_name), "correlation_heatmap.png")

    df = pd.DataFrame({
        "True Values": true_values,
        "Predicted Values": predicted_values
    })
    corr = df.corr()

    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Between True and Predicted Values")

    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def plot_training_progress(losses, model_name):
    """
    Displays the evolution of the loss function during training (if available) 
    and saves it in the model's folder.
    """
    if not losses:
        return

    save_path = os.path.join(ensure_model_dir(model_name), "training_progress.png")

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=np.arange(len(losses)), y=losses, color="green")
    plt.title("Training Loss Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()

#Plots for the Trading Strategies  

# SL_STRATEGYS

def plot_trading_strategy_performance(strategies, model_name):
    """
    Plots comparative performance of multiple SL trading strategies
    (pure forecasting, mean reversion, hybrid, ground truth).

    Each strategy object should be an instance of SLTradingStrategy after simulation.
    """
    model_dir = ensure_model_dir(model_name)

    # Extract strategy names and thresholds
    strategy_names = [s.strategy_name for s in strategies]
    thresholds = strategies[0].trade_thresholds

    # Define a distinct color palette for each strategy
    color_map = {
        "pure forcasting": "#1f77b4",     # strong blue
        "mean reversion": "#ff7f0e",      # vivid orange
        "hybrid": "#2ca02c",              # rich green
        "threshold_based_strategy": "#d62728"         # bright red
    }

    # ---- Total Profit per Strategy ----
    plt.figure(figsize=(10, 6))
    for s in strategies:
        sns.lineplot(
            x=thresholds,
            y=s.total_profit_or_loss,
            label=s.strategy_name,
            color=color_map.get(s.strategy_name, None),
            linewidth=2.5
        )
    plt.title("Total Profit vs Trade Threshold")
    plt.xlabel("Trade Threshold")
    plt.ylabel("Total Profit")
    plt.legend()
    save_path = os.path.join(model_dir, "total_profit_vs_threshold.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()

    # ---- Sharpe Ratios per Strategy ----
    plt.figure(figsize=(10, 6))
    for s in strategies:
        sns.lineplot(
            x=thresholds,
            y=s.sharpe_ratios,
            label=s.strategy_name,
            color=color_map.get(s.strategy_name, None),
            linewidth=2.5
        )
    plt.title("Sharpe Ratio vs Trade Threshold")
    plt.xlabel("Trade Threshold")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    save_path = os.path.join(model_dir, "sharpe_ratio_vs_threshold.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()

    # ---- Number of Trades ----
    plt.figure(figsize=(10, 6))
    for s in strategies:
        sns.lineplot(
            x=thresholds,
            y=s.num_trade,
            label=f"{s.strategy_name} (executed)",
            color=color_map.get(s.strategy_name, None),
            linewidth=2.5
        )
        sns.lineplot(
            x=thresholds,
            y=s.no_trade,
            linestyle="--",
            label=f"{s.strategy_name} (skipped)",
            color=color_map.get(s.strategy_name, None),
            linewidth=1.8
        )
    plt.title("Number of Trades vs Trade Threshold")
    plt.xlabel("Trade Threshold")
    plt.ylabel("Number of Trades")
    plt.legend()
    save_path = os.path.join(model_dir, "num_trades_vs_threshold.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()

    # ---- Profit per Trade ----
    plt.figure(figsize=(10, 6))
    for s in strategies:
        profit_per_trade = [
            (total / num) if num > 0 else 0
            for total, num in zip(s.total_profit_or_loss, s.num_trade)
        ]
        sns.lineplot(
            x=thresholds,
            y=profit_per_trade,
            label=s.strategy_name,
            color=color_map.get(s.strategy_name, None),
            linewidth=2.5
        )
    plt.title("Profit per Trade vs Trade Threshold")
    plt.xlabel("Trade Threshold")
    plt.ylabel("Average Profit per Trade")
    plt.legend()
    save_path = os.path.join(model_dir, "profit_per_trade_vs_threshold.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()

def plot_confusion_matrices(strategies, model_name):
    """
    Plots confusion matrices for each SL trading strategy and threshold.
    Each matrix shows TP, FP, FN, TN (and optionally No Change).
    """
    model_dir = ensure_model_dir(model_name)

    for s in strategies:
        thresholds = s.trade_thresholds

        for j, cm in enumerate(s.confusion_matrices):
            # Build a simple 2x2 confusion matrix for visualization
            matrix = np.array([
                [cm.true_positive, cm.false_negative],
                [cm.false_positive, cm.true_negative]
            ])

            labels = ["Positive", "Negative"]
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="YlGnBu",
                xticklabels=labels,
                yticklabels=labels
            )

            plt.title(
                f"Confusion Matrix - {s.strategy_name}\nThreshold = {thresholds[j]}"
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            save_path = os.path.join(
                model_dir,
                f"confusion_matrix_{s.strategy_name.replace(' ', '_')}_th{j}.png"
            )
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
            plt.close()

            # Optional: show no_change count separately
            if hasattr(cm, "no_change") and cm.no_change > 0:
                plt.figure(figsize=(3, 3))
                sns.heatmap(
                    np.array([[cm.no_change]]),
                    annot=True,
                    fmt="d",
                    cmap="Purples",
                    cbar=False
                )
                plt.title(
                    f"No-Change Count - {s.strategy_name}\nThreshold = {thresholds[j]}"
                )
                save_path_nc = os.path.join(
                    model_dir,
                    f"no_change_{s.strategy_name.replace(' ', '_')}_th{j}.png"
                )
                plt.savefig(save_path_nc, bbox_inches="tight")
                print(f"Plot saved to: {save_path_nc}")
                plt.close()

def plot_accuracy(strategies, model_name):
    """
    Calculates and plots accuracy for each SL trading strategy across thresholds.
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    """

    model_dir = ensure_model_dir(model_name)

    color_map = {
        "pure forcasting": "#1f77b4",     # strong blue
        "mean reversion": "#ff7f0e",      # vivid orange
        "hybrid": "#2ca02c",              # rich green
        "threshold_based_strategy": "#d62728"  # bright red
    }

    plt.figure(figsize=(10, 6))
    for s in strategies:
        accuracies = []
        for cm in s.confusion_matrices:
            total = cm.true_positive + cm.true_negative + cm.false_positive + cm.false_negative
            accuracy = (cm.true_positive + cm.true_negative) / total if total > 0 else 0
            accuracies.append(accuracy)
        sns.lineplot(
            x=s.trade_thresholds,
            y=accuracies,
            label=s.strategy_name,
            linewidth=2.5,
            color=color_map.get(s.strategy_name, None)
        )

    plt.title("Accuracy vs Trade Threshold")
    plt.xlabel("Trade Threshold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()

    save_path = os.path.join(model_dir, "accuracy_vs_threshold.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to: {save_path}\n")
    plt.close()

