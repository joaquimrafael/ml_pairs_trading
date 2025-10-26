import pickle
import os
from .graphics import ensure_model_dir


def save_predictions(true_values, predicted_values, model_name):
    """Saves the true and predicted values arrays in the model's folder."""
    dir_path = ensure_model_dir(model_name)
    path = os.path.join(dir_path, "predictions.pkl")
    with open(path, "wb") as f:
        pickle.dump({"true_values": true_values, "predicted_values": predicted_values}, f)
    print(f"Predictions saved to: {path}")

def save_metrics_df(metrics_df, model_name):
    """Saves the metrics DataFrame (accuracy, precision, recall, f1) to CSV."""
    dir_path = ensure_model_dir(model_name)
    path = os.path.join(dir_path, "metrics.csv")
    metrics_df.to_csv(path, index=False)
    print(f"Metrics saved to: {path}")

def save_strategy_results(strategies, model_name):
    """Saves the SL strategies results as a pickle file."""
    dir_path = ensure_model_dir(model_name)
    results = {}
    for s in strategies:
        results[s.strategy_name] = {
            "total_profit": s.total_profit_or_loss,
            "sharpe_ratios": s.sharpe_ratios,
            "num_trade": s.num_trade,
            "no_trade": s.no_trade,
        }
    path = os.path.join(dir_path, "strategies.pkl")
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Strategy results saved to: {path}")