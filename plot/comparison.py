import os
import pickle
import math
import argparse
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==== Styling ====
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# =========================================================
# I/O helpers
# =========================================================
def find_model_dirs(base_dir: str, dataset_name: str) -> List[str]:
    """Find all model folders inside the dataset folder."""
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory {dataset_dir} not found.")

    model_dirs = []
    for root, dirs, files in os.walk(dataset_dir):
        if any(f in files for f in ["metrics.csv", "strategies.pkl", "predictions.pkl"]):
            if root not in model_dirs:
                model_dirs.append(root)
    # Only keep direct subfolders (model_name folders)
    # but allow flexible depth if your structure has an extra level
    return sorted(set(model_dirs))

def load_model_data(model_dir: str) -> Dict[str, Any]:
    """Load saved files for a single model directory."""
    data = {}

    metrics_path = os.path.join(model_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        try:
            data["metrics"] = pd.read_csv(metrics_path)
        except Exception:
            pass

    predictions_path = os.path.join(model_dir, "predictions.pkl")
    if os.path.exists(predictions_path):
        with open(predictions_path, "rb") as f:
            data["predictions"] = pickle.load(f)

    strategies_path = os.path.join(model_dir, "strategies.pkl")
    if os.path.exists(strategies_path):
        with open(strategies_path, "rb") as f:
            data["strategies"] = pickle.load(f)

    return data

def aggregate_models_for_dataset(base_dir: str, dataset_name: str,
                                 models_filter: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Load data from all models for a dataset; optionally filter model names."""
    model_dirs = find_model_dirs(base_dir, dataset_name)
    if not model_dirs:
        print(f"No model directories found for dataset '{dataset_name}' in '{base_dir}'.")
        return {}

    all_models_data = {}
    for dir_path in model_dirs:
        model_name = os.path.basename(dir_path)
        if models_filter and model_name not in models_filter:
            continue
        all_models_data[model_name] = load_model_data(dir_path)
    return all_models_data

# =========================================================
# Metric computation from predictions
# =========================================================
def _to_np(x) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    return arr.astype(float)

def compute_basic_metrics(true_values: np.ndarray, pred_values: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, sMAPE, R2, Directional Accuracy."""
    eps = 1e-12
    err = pred_values - true_values
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))

    # MAPE: ignore zero true values
    mask_nonzero = np.abs(true_values) > eps
    mape = float(np.mean(np.abs(err[mask_nonzero] / (true_values[mask_nonzero] + eps)))) if mask_nonzero.any() else np.nan

    # sMAPE
    smape = float(np.mean(2.0 * abs_err / (np.abs(true_values) + np.abs(pred_values) + eps)))

    # R^2
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true_values - np.mean(true_values))**2)) + eps
    r2 = 1.0 - ss_res / ss_tot

    # Directional Accuracy (sign of first difference)
    # Compare direction between consecutive steps
    if len(true_values) > 1:
        true_dir = np.sign(np.diff(true_values))
        pred_dir = np.sign(np.diff(pred_values))
        da = float(np.mean(true_dir == pred_dir))
    else:
        da = np.nan

    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "r2": r2, "directional_accuracy": da}

def extract_predictions(model_data: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (true, pred) if available and aligned, else None."""
    if "predictions" not in model_data:
        return None
    preds = model_data["predictions"]
    if not ("true_values" in preds and "predicted_values" in preds):
        return None
    true_values = _to_np(preds["true_values"])
    pred_values = _to_np(preds["predicted_values"])
    # Align size (just in case)
    n = min(true_values.size, pred_values.size)
    if n == 0:
        return None
    return true_values[:n], pred_values[:n]

def build_metrics_table(all_models_data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Compute metrics for every model using predictions.pkl (preferred for ML comparison)."""
    rows = []
    for model_name, data in all_models_data.items():
        pair = extract_predictions(data)
        if pair is None:
            continue
        true_values, pred_values = pair
        m = compute_basic_metrics(true_values, pred_values)
        m["model"] = model_name
        rows.append(m)
    return pd.DataFrame(rows).set_index("model").sort_index()

# =========================================================
# Diebold–Mariano test (simple HAC / Newey–West variance)
# =========================================================
def newey_west_variance(d: np.ndarray, max_lag: Optional[int] = None) -> float:
    """HAC variance estimate for loss differential series d_t."""
    T = len(d)
    if T <= 2:
        return float(np.var(d, ddof=1) / max(T, 1))
    if max_lag is None:
        max_lag = int(1.5 * (T ** (1/3)))  # common heuristic
    d = d - np.mean(d)
    gamma0 = np.sum(d * d) / T
    var = gamma0
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        cov = np.sum(d[lag:] * d[:-lag]) / T
        var += 2.0 * w * cov
    return var

def dm_test(true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, loss: str = "SE") -> Tuple[float, float]:
    """
    Diebold–Mariano test between model A and B.
    loss: "SE" -> squared error, "AE" -> absolute error.
    Returns (DM statistic, two-sided p-value approx using N(0,1)).
    """
    n = min(len(true), len(pred_a), len(pred_b))
    y, a, b = true[:n], pred_a[:n], pred_b[:n]
    if loss.upper() == "SE":
        la = (y - a) ** 2
        lb = (y - b) ** 2
    else:
        la = np.abs(y - a)
        lb = np.abs(y - b)
    d = la - lb
    mean_d = float(np.mean(d))
    var_d = newey_west_variance(d)
    if var_d <= 0:
        return np.nan, np.nan
    stat = mean_d / math.sqrt(var_d / n)
    # p-value from normal approx
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(stat) / math.sqrt(2))))
    return float(stat), float(p)

# =========================================================
# Plotting
# =========================================================
def savefig(path: str, title: Optional[str] = None, legend_title: Optional[str] = None):
    if title:
        plt.title(title)
    if legend_title:
        leg = plt.gca().get_legend()
        if leg is not None:
            leg.set_title(legend_title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_metric_bars(metrics_df: pd.DataFrame, outdir: str):
    """Bar plots of key metrics per model."""
    display_cols = ["mae", "rmse", "mape", "smape", "r2", "directional_accuracy"]
    for col in display_cols:
        if col not in metrics_df.columns:
            continue
        plt.figure(figsize=(10, 6))
        sns.barplot(x=metrics_df.index, y=metrics_df[col], palette="tab10")
        plt.xlabel("Model")
        plt.ylabel(col.upper() if col != "r2" else "R²")
        title = col.replace("_", " ").title()
        savefig(os.path.join(outdir, f"{col}_by_model.png"), title=title)

def plot_calibration(all_models_data: Dict[str, Dict[str, Any]], outdir: str, max_points: int = 5000):
    """Scatter of True vs Predicted with 45-degree line for each model."""
    plt.figure(figsize=(8, 8))
    for model_name, data in all_models_data.items():
        pair = extract_predictions(data)
        if pair is None:
            continue
        y, p = pair
        n = len(y)
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points).astype(int)
            y, p = y[idx], p[idx]
        sns.scatterplot(x=y, y=p, s=12, alpha=0.35, label=model_name)
    # identity line
    lims = plt.xlim()
    plt.plot(lims, lims, color="black", linewidth=1.5, linestyle="--", label="Ideal (45°)")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    savefig(os.path.join(outdir, "calibration_true_vs_pred.png"), title="Calibration: True vs Predicted", legend_title="Model")

def plot_residual_boxplot(all_models_data: Dict[str, Dict[str, Any]], outdir: str):
    """Boxplot of residuals per model."""
    rows = []
    for model_name, data in all_models_data.items():
        pair = extract_predictions(data)
        if pair is None:
            continue
        y, p = pair
        err = (p - y).reshape(-1)
        rows.append(pd.DataFrame({"model": model_name, "residual": err}))
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="model", y="residual", data=df, palette="tab10", showfliers=False)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1)
    savefig(os.path.join(outdir, "residuals_boxplot.png"), title="Residuals by Model")

def plot_rolling_rmse(all_models_data: Dict[str, Dict[str, Any]], outdir: str, window: int = 500):
    """Rolling RMSE over time for each model (aligned window)."""
    plt.figure(figsize=(12, 6))
    for model_name, data in all_models_data.items():
        pair = extract_predictions(data)
        if pair is None:
            continue
        y, p = pair
        e2 = (p - y) ** 2
        if len(e2) < window:
            continue
        rmse_roll = np.sqrt(pd.Series(e2).rolling(window).mean()).values
        sns.lineplot(x=np.arange(len(rmse_roll)), y=rmse_roll, label=model_name)
    plt.xlabel("Step")
    plt.ylabel(f"Rolling RMSE (window={window})")
    savefig(os.path.join(outdir, f"rolling_rmse_w{window}.png"), title="Rolling RMSE", legend_title="Model")

def plot_directional_accuracy_bar(metrics_df: pd.DataFrame, outdir: str):
    """Bar plot of Directional Accuracy by model."""
    if "directional_accuracy" not in metrics_df.columns:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metrics_df.index, y=metrics_df["directional_accuracy"], palette="tab10")
    plt.ylim(0, 1)
    plt.ylabel("Directional Accuracy")
    savefig(os.path.join(outdir, "directional_accuracy_by_model.png"), title="Directional Accuracy by Model")

def plot_dm_against_baseline(all_models_data: Dict[str, Dict[str, Any]], outdir: str,
                             baseline: str, loss: str = "SE"):
    """
    Diebold–Mariano test against a chosen baseline model.
    Bar plot of p-values (lower => significantly better than baseline).
    """
    if baseline not in all_models_data:
        print(f"[DM] Baseline '{baseline}' not found. Skipping DM plots.")
        return

    base_pair = extract_predictions(all_models_data[baseline])
    if base_pair is None:
        print(f"[DM] Baseline '{baseline}' has no predictions. Skipping DM plots.")
        return

    y_base, p_base = base_pair
    pvals = []
    labels = []
    for model_name, data in all_models_data.items():
        if model_name == baseline:
            continue
        pair = extract_predictions(data)
        if pair is None:
            continue
        y, p = pair
        n = min(len(y_base), len(y))
        stat, pval = dm_test(y[:n], p[:n], p_base[:n], loss=loss)
        if not np.isnan(pval):
            labels.append(model_name)
            pvals.append(pval)

    if not labels:
        print("[DM] No comparable models for DM test.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=pvals, palette="tab10")
    plt.axhline(0.05, color="red", linestyle="--", linewidth=1, label="α = 0.05")
    savefig(os.path.join(outdir, f"dm_pvalues_vs_{baseline}.png"),
            title=f"Diebold–Mariano p-values vs '{baseline}'", legend_title=None)

# =========================================================
# Legacy (optional): strategy plots kept if strategies.pkl exists
# =========================================================
def plot_strategy_profits(all_models_data: Dict[str, Dict[str, Any]], save_path: str = "."):
    """
    Optional: compare total profits / Sharpe per strategy across models.
    Useful context, but secondary when the goal is *model* comparison.
    """
    df_list = []
    for model_name, data in all_models_data.items():
        if 'strategies' in data:
            strategies = data['strategies']
            for strategy_name, s_data in strategies.items():
                df_list.append({
                    "model": model_name,
                    "strategy": strategy_name,
                    "total_profit": float(np.sum(s_data.get('total_profit', [0]))),
                    "sharpe_ratio": float(np.max(s_data.get('sharpe_ratios', [0]))) if s_data.get('sharpe_ratios') else 0,
                    "num_trades": int(np.sum(s_data.get('num_trade', [0])))
                })
    if not df_list:
        print("No 'strategies.pkl' files found to plot (strategy comparison skipped).")
        return

    df = pd.DataFrame(df_list)

    # Total Profit
    plt.figure(figsize=(12, 6))
    sns.barplot(x="strategy", y="total_profit", hue="model", data=df)
    plt.xticks(rotation=15, ha='right')
    savefig(os.path.join(save_path, "total_profit_comparison.png"),
            title="Total Profit by Strategy and Model", legend_title="Model")

    # Sharpe Ratio
    plt.figure(figsize=(12, 6))
    sns.barplot(x="strategy", y="sharpe_ratio", hue="model", data=df)
    plt.xticks(rotation=15, ha='right')
    savefig(os.path.join(save_path, "sharpe_ratio_comparison.png"),
            title="Sharpe Ratio by Strategy and Model", legend_title="Model")

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Compare ML models on the same dataset (prediction quality focus).")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name folder (e.g., itau3_4_min).")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory where dataset folders are located.")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated list of model folders to include (optional).")
    parser.add_argument("--baseline_model", type=str, default="",
                        help="If set, run DM test against this model (name must match folder).")
    parser.add_argument("--dm_loss", type=str, default="SE", choices=["SE", "AE"],
                        help="Loss for DM test: SE (squared error) or AE (absolute error).")
    parser.add_argument("--rolling_window", type=int, default=500,
                        help="Rolling window size for rolling RMSE plot.")
    args = parser.parse_args()

    models_filter = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None

    all_models_data = aggregate_models_for_dataset(args.base_dir, args.dataset, models_filter=models_filter)
    if not all_models_data:
        return

    # Output dir for plots
    outdir = os.path.join(args.base_dir, args.dataset, "comparison_plots_ml")
    os.makedirs(outdir, exist_ok=True)
    print(f"Plots will be saved in: {outdir}")

    # Build metrics table from predictions
    metrics_df = build_metrics_table(all_models_data)
    if metrics_df.empty:
        print("No predictions found to compute ML metrics. Exiting.")
        return

    # Save metrics table
    metrics_csv = os.path.join(outdir, "model_metrics_summary.csv")
    metrics_df.to_csv(metrics_csv)
    print(f"Saved metrics summary: {metrics_csv}")

    # Plots focused on ML model comparison
    plot_metric_bars(metrics_df, outdir)
    plot_calibration(all_models_data, outdir)
    plot_residual_boxplot(all_models_data, outdir)
    plot_rolling_rmse(all_models_data, outdir, window=args.rolling_window)
    plot_directional_accuracy_bar(metrics_df, outdir)

    # Optional: DM test vs baseline
    if args.baseline_model:
        plot_dm_against_baseline(all_models_data, outdir, baseline=args.baseline_model, loss=args.dm_loss)

    # Optional legacy context: strategy comparison (secondary)
    plot_strategy_profits(all_models_data, save_path=outdir)

if __name__ == "__main__":
    main()
