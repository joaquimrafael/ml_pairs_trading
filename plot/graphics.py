import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# General Seaborn style configuration
sns.set_theme(style="whitegrid", palette="crest")

def ensure_model_dir(model_name):
    """
    Ensures that the directory 'plot/<model_name>/' exists.
    Returns the path to the model's plot directory.
    """
    model_dir = os.path.join("plot", model_name)
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
