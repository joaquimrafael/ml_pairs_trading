import os
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Global seaborn theme (only for grids/fonts); we won't rely on seaborn for critical lines.
sns.set_theme(style="whitegrid", palette="crest")

# ---------------------------
# Fixed, colorblind-friendly strategy colors (Okabe–Ito inspired)
# ---------------------------
_STRATEGY_COLORS = {
    "pure forcasting": "#0072B2",  # blue  (kept exact name to match your codebase)
    "mean reversion":  "#E69F00",  # orange
    "hybrid":           "#009E73",  # green
    "momentum":         "#D55E00",  # vermillion/red
    "ground truth":     "#CC79A7",  # purple
}

# Fallback palette for unexpected strategy names (deterministic)
_FALLBACK_PALETTE = [
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # gray
    "#000000",  # black
    "#4E79A7",  # blue-alt
    "#59A14F",  # green-alt
    "#E15759",  # red-alt
    "#B07AA1",  # purple-alt
]
_ASSIGNED_FALLBACKS = {}  # name -> index

# Deterministic markers per strategy (helps visual separation)
_STRATEGY_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
_MARKER_CACHE = {}  # name -> marker index

# Fixed styles for classification metrics (consistent across plots)
_METRIC_STYLES = {
    "Accuracy":  {"color": "#000000", "linestyle": "-",  "marker": "o"},
    "Precision": {"color": "#6A3D9A", "linestyle": "--", "marker": "s"},
    "Recall":    {"color": "#1B9E77", "linestyle": "-.", "marker": "D"},
    "F1-Score":  {"color": "#E31A1C", "linestyle": ":",  "marker": "^"},
}

# ---------------------------
# Dataset selection (used to build output dir)
# ---------------------------
data_set = ""

def set_data_set(data_set_name: str):
    """Choose dataset name (used to build plot directories)."""
    global data_set
    data_set = data_set_name

def ensure_model_dir(model_name: str) -> str:
    """
    Ensures that the directory 'plot/<dataset_name>/<model_name>/' exists.
    Returns the path to the model's plot directory.
    """
    dataset_name = os.path.splitext(os.path.basename(data_set))[0]
    model_dir = os.path.join("plot", dataset_name, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

# ---------------------------
# Internal helpers
# ---------------------------
def _color_for(name: str) -> str:
    """Return a consistent, high-contrast color for a given strategy name."""
    if name in _STRATEGY_COLORS:
        return _STRATEGY_COLORS[name]
    if name not in _ASSIGNED_FALLBACKS:
        _ASSIGNED_FALLBACKS[name] = len(_ASSIGNED_FALLBACKS) % len(_FALLBACK_PALETTE)
    return _FALLBACK_PALETTE[_ASSIGNED_FALLBACKS[name]]

def _marker_for(name: str) -> str:
    """Deterministic marker per strategy name."""
    if name not in _MARKER_CACHE:
        _MARKER_CACHE[name] = len(_MARKER_CACHE) % len(_STRATEGY_MARKERS)
    return _STRATEGY_MARKERS[_MARKER_CACHE[name]]

def _finalize_and_save(path: str, title: str | None = None, legend_title: str | None = None, show_legend: bool = True):
    """
    Standard finishing: optionally add legend (and title), tight layout, save, close.
    Now it CREATES the legend if there are labeled artists.
    """
    ax = plt.gca()

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend()
            if legend_title and leg is not None:
                leg.set_title(legend_title)

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Plot saved to: {path}")
    plt.close()

def _axhline_zero():
    """Zero line reference for PnL plots."""
    plt.axhline(0, linewidth=1, color="#888888", alpha=0.9)

def _plot_line(x, y, label, color, marker, linestyle="-", linewidth=2.6):
    """
    Matplotlib-only line plotting to avoid seaborn smoothing / transparency quirks.
    """
    plt.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=linewidth)

# ---------------------------
# Model-level plots (complementary)
# ---------------------------
def plot_model_metrics(metrics_dict, model_name, title="Evaluation Metrics"):
    """Bar chart with aggregated model error metrics (complementary)."""
    save_path = os.path.join(ensure_model_dir(model_name), "metrics_plot.png")
    plt.figure(figsize=(8, 5))
    bars = plt.bar(list(metrics_dict.keys()), list(metrics_dict.values()), color="#56B4E9")
    plt.ylabel('Error')
    vmax = max(metrics_dict.values()) if metrics_dict else 0
    for i, v in enumerate(metrics_dict.values()):
        plt.text(i, v + (0.01 * vmax if vmax != 0 else 0.01), f"{v:.4f}", ha='center')
    _finalize_and_save(save_path, title=title, show_legend=False)

def plot_predictions(true_values, predicted_values, model_name):
    """Time series of true vs predicted values."""
    save_path = os.path.join(ensure_model_dir(model_name), "predictions_vs_true.png")
    plt.figure(figsize=(10, 5))
    x_true = np.arange(len(true_values))
    x_pred = np.arange(len(predicted_values))
    _plot_line(x_true, true_values, "True",  "#0072B2", "o", "-")
    _plot_line(x_pred, predicted_values, "Predicted", "#D55E00", "s", "-")
    plt.xlabel("Observation")
    plt.ylabel("Ratio")
    _finalize_and_save(save_path, title="True vs Predicted Values", legend_title="Series")

def plot_error_distribution(true_values, predicted_values, model_name):
    """Distribution of prediction errors = (pred - true) (complementary)."""
    save_path = os.path.join(ensure_model_dir(model_name), "error_distribution.png")
    errors = np.array(predicted_values) - np.array(true_values)
    plt.figure(figsize=(8, 5))
    sns.histplot(errors, kde=True, bins=30, color="#E69F00")
    plt.xlabel("Error (Predicted - True)")
    plt.ylabel("Frequency")
    _finalize_and_save(save_path, title="Prediction Error Distribution", show_legend=False)

def plot_correlation_heatmap(true_values, predicted_values, model_name):
    """Correlation between true and predicted values (complementary)."""
    save_path = os.path.join(ensure_model_dir(model_name), "correlation_heatmap.png")
    df = pd.DataFrame({"True Values": true_values, "Predicted Values": predicted_values})
    corr = df.corr()
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    _finalize_and_save(save_path, title="Correlation Between True and Predicted Values", show_legend=False)

def plot_training_progress(losses, model_name):
    """Training loss across epochs (only if you log training history)."""
    if not losses:
        return
    save_path = os.path.join(ensure_model_dir(model_name), "training_progress.png")
    plt.figure(figsize=(8, 5))
    _plot_line(np.arange(len(losses)), losses, label="Loss", color="#000000", marker="o", linestyle="-", linewidth=2.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    _finalize_and_save(save_path, title="Training Loss Progress")

# ---------------------------
# Strategy-level plots (core for comparison)
# ---------------------------
def plot_trading_strategy_performance(strategies, model_name):
    """
    Core comparison across strategies:
      - Total Profit vs Threshold
      - Sharpe Ratio vs Threshold
      - (Executed vs Skipped) Number of Trades vs Threshold
      - Profit per Trade vs Threshold
    """
    model_dir = ensure_model_dir(model_name)
    thresholds = strategies[0].trade_thresholds  # assume aligned

    # Total Profit
    plt.figure(figsize=(10, 6))
    for s in strategies:
        _plot_line(thresholds, s.total_profit_or_loss, s.strategy_name,
                   _color_for(s.strategy_name), _marker_for(s.strategy_name))
    _axhline_zero()
    plt.xlabel("Trade Threshold")
    plt.ylabel("Total Profit")
    _finalize_and_save(os.path.join(model_dir, "total_profit_vs_threshold.png"),
                       title="Total Profit vs Trade Threshold",
                       legend_title="Strategy")

    # Sharpe Ratio
    plt.figure(figsize=(10, 6))
    for s in strategies:
        _plot_line(thresholds, s.sharpe_ratios, s.strategy_name,
                   _color_for(s.strategy_name), _marker_for(s.strategy_name))
    plt.xlabel("Trade Threshold")
    plt.ylabel("Sharpe Ratio")
    _finalize_and_save(os.path.join(model_dir, "sharpe_ratio_vs_threshold.png"),
                       title="Sharpe Ratio vs Trade Threshold",
                       legend_title="Strategy")

    # Number of Trades (executed vs skipped)
    plt.figure(figsize=(10, 6))
    for s in strategies:
        _plot_line(thresholds, s.num_trade, f"{s.strategy_name} – executed",
                   _color_for(s.strategy_name), _marker_for(s.strategy_name), "-")
        _plot_line(thresholds, s.no_trade, f"{s.strategy_name} – skipped",
                   _color_for(s.strategy_name), _marker_for(s.strategy_name), "--", linewidth=2.0)
    plt.xlabel("Trade Threshold")
    plt.ylabel("Count")
    _finalize_and_save(os.path.join(model_dir, "num_trades_vs_threshold.png"),
                       title="Number of Trades vs Trade Threshold",
                       legend_title="Series")

    # Profit per Trade
    plt.figure(figsize=(10, 6))
    for s in strategies:
        profit_per_trade = [(total / num) if num > 0 else 0
                            for total, num in zip(s.total_profit_or_loss, s.num_trade)]
        _plot_line(thresholds, profit_per_trade, s.strategy_name,
                   _color_for(s.strategy_name), _marker_for(s.strategy_name))
    _axhline_zero()
    plt.xlabel("Trade Threshold")
    plt.ylabel("Average Profit per Trade")
    _finalize_and_save(os.path.join(model_dir, "profit_per_trade_vs_threshold.png"),
                       title="Profit per Trade vs Trade Threshold",
                       legend_title="Strategy")

def plot_confusion_matrices(strategies, model_name):
    """
    Combined confusion matrices: one PNG per strategy with a grid of 2x2 matrices
    (one per threshold). Highlights the best threshold by Sharpe in the subplot title.
    """
    model_dir = ensure_model_dir(model_name)

    for s in strategies:
        thresholds = s.trade_thresholds
        cms = s.confusion_matrices
        n = len(cms)
        if n == 0:
            continue

        cols = min(5, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows))

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        best_idx = None
        if getattr(s, "sharpe_ratios", None):
            try:
                best_idx = int(np.argmax(s.sharpe_ratios))
            except Exception:
                best_idx = None

        for j, cm in enumerate(cms):
            r, c = divmod(j, cols)
            ax = axes[r, c]

            matrix = np.array([
                [cm.true_positive, cm.false_negative],
                [cm.false_positive, cm.true_negative]
            ])

            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="YlGnBu",
                xticklabels=["Pred +", "Pred -"],
                yticklabels=["Actual +", "Actual -"],
                cbar=False,
                ax=ax
            )

            th = thresholds[j] if j < len(thresholds) else j
            title = f"th={th}"
            if best_idx is not None and j == best_idx:
                title += "  (best Sharpe)"
            ax.set_title(title, fontsize=10)

        # Hide unused axes
        for k in range(n, rows * cols):
            r, c = divmod(k, cols)
            axes[r, c].axis("off")

        plt.suptitle(f"Confusion Matrices — {s.strategy_name}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = os.path.join(model_dir, f"confusion_matrices_{s.strategy_name.replace(' ', '_')}.png")
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        plt.close()

def plot_confusion_overview_best(strategies, model_name, criterion="sharpe"):
    """
    Overview with ONE 2x2 matrix per strategy (best threshold by 'sharpe' or 'profit').
    """
    assert criterion in ("sharpe", "profit")
    model_dir = ensure_model_dir(model_name)

    kept = []
    for s in strategies:
        if criterion == "sharpe":
            if not getattr(s, "sharpe_ratios", None):
                continue
            idx = int(np.argmax(s.sharpe_ratios))
        else:
            idx = int(np.argmax(s.total_profit_or_loss))
        if idx < len(s.confusion_matrices):
            kept.append((s, idx))

    if not kept:
        print("No strategies available for confusion overview.")
        return

    cols = min(5, len(kept))
    rows = int(np.ceil(len(kept) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, (s, idx) in enumerate(kept):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        cm = s.confusion_matrices[idx]
        matrix = np.array([
            [cm.true_positive, cm.false_negative],
            [cm.false_positive, cm.true_negative]
        ])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=["Pred +", "Pred -"],
            yticklabels=["Actual +", "Actual -"],
            cbar=False,
            ax=ax
        )
        th = s.trade_thresholds[idx] if idx < len(s.trade_thresholds) else idx
        ax.set_title(f"{s.strategy_name}\nth={th}", fontsize=10)

    for k in range(len(kept), rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    plt.suptitle(f"Confusion Overview — best {criterion}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(model_dir, f"confusion_overview_best_{criterion}.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()

def plot_accuracy(strategies, model_name):
    """
    Accuracy across thresholds for each strategy.
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    """
    model_dir = ensure_model_dir(model_name)
    plt.figure(figsize=(10, 6))
    for s in strategies:
        accuracies = []
        for cm in s.confusion_matrices:
            total = cm.true_positive + cm.true_negative + cm.false_positive + cm.false_negative
            accuracies.append((cm.true_positive + cm.true_negative) / total if total > 0 else 0.0)
        _plot_line(s.trade_thresholds, accuracies, s.strategy_name,
                   _color_for(s.strategy_name), _marker_for(s.strategy_name))
    plt.xlabel("Trade Threshold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    _finalize_and_save(os.path.join(model_dir, "accuracy_vs_threshold.png"),
                       title="Accuracy vs Trade Threshold",
                       legend_title="Strategy")

def plot_model_performance_curves(metrics_df, model_name):
    """
    Multi-metric (Accuracy, Precision, Recall, F1) vs threshold from a single metrics_df.
    Colors/linestyles are FIXED to avoid any similarity.
    """
    model_dir = ensure_model_dir(model_name)
    save_path = os.path.join(model_dir, "model_performance_curves.png")

    # Normalize column names once
    norm_cols = {c.lower().replace("-", "").replace(" ", ""): c for c in metrics_df.columns}
    plt.figure(figsize=(10, 6))
    for metric, style in _METRIC_STYLES.items():
        key = metric.lower().replace("-", "").replace(" ", "")
        if key in norm_cols:
            y = metrics_df[norm_cols[key]]
            _plot_line(metrics_df['threshold'], y, metric,
                       style["color"], style["marker"], style["linestyle"], linewidth=2.6)
    plt.xlabel("Trade Threshold")
    plt.ylabel("Metric Value")
    plt.ylim(0, 1)
    _finalize_and_save(save_path, title="Model Performance vs Threshold", legend_title="Metric")

def plot_strategy_classification_curves(strategies, model_name):
    """
    For each strategy, plot Accuracy/Precision/Recall/F1 vs threshold based on its confusion matrices.
    Uses FIXED colors/linestyles for metrics to avoid similar colors.
    """
    model_dir = ensure_model_dir(model_name)

    for s in strategies:
        thresholds = s.trade_thresholds
        accs, precs, recs, f1s = [], [], [], []
        for cm in s.confusion_matrices:
            tp, fp, fn, tn = cm.true_positive, cm.false_positive, cm.false_negative, cm.true_negative
            total = tp + fp + fn + tn
            acc = (tp + tn) / total if total > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

        plt.figure(figsize=(10, 6))
        _plot_line(thresholds, accs, 'Accuracy',
                   _METRIC_STYLES["Accuracy"]["color"], _METRIC_STYLES["Accuracy"]["marker"],
                   _METRIC_STYLES["Accuracy"]["linestyle"])
        _plot_line(thresholds, precs, 'Precision',
                   _METRIC_STYLES["Precision"]["color"], _METRIC_STYLES["Precision"]["marker"],
                   _METRIC_STYLES["Precision"]["linestyle"])
        _plot_line(thresholds, recs, 'Recall',
                   _METRIC_STYLES["Recall"]["color"], _METRIC_STYLES["Recall"]["marker"],
                   _METRIC_STYLES["Recall"]["linestyle"])
        _plot_line(thresholds, f1s, 'F1-Score',
                   _METRIC_STYLES["F1-Score"]["color"], _METRIC_STYLES["F1-Score"]["marker"],
                   _METRIC_STYLES["F1-Score"]["linestyle"])
        plt.xlabel("Trade Threshold")
        plt.ylabel("Metric Value")
        plt.ylim(0, 1)
        _finalize_and_save(os.path.join(model_dir, f"classification_curves_{s.strategy_name.replace(' ', '_')}.png"),
                           title=f"{s.strategy_name} — Classification Metrics vs Threshold",
                           legend_title="Metric")

def plot_cumulative_pnl_best_threshold(strategies, model_name, criterion="sharpe"):
    """
    Cumulative P&L over time for each strategy using the best threshold by:
       - 'sharpe' (max Sharpe), or
       - 'profit' (max Total Profit).
    Colors and markers are FIXED per strategy. Uses matplotlib-only lines to avoid seaborn artifacts.
    """
    assert criterion in ("sharpe", "profit")
    model_dir = ensure_model_dir(model_name)

    plt.figure(figsize=(12, 6))
    for s in strategies:
        if criterion == "sharpe":
            if not s.sharpe_ratios:
                continue
            best_idx = int(np.argmax(s.sharpe_ratios))
        else:
            best_idx = int(np.argmax(s.total_profit_or_loss))

        step_pnl = s.all_profit_or_loss[best_idx] if best_idx < len(s.all_profit_or_loss) else []
        if len(step_pnl) == 0:
            continue
        cum_pnl = np.cumsum(step_pnl)

        _plot_line(
            np.arange(len(cum_pnl)),
            cum_pnl,
            label=f"{s.strategy_name} (best {criterion})",
            color=_color_for(s.strategy_name),
            marker=_marker_for(s.strategy_name),
            linestyle="-",
            linewidth=2.8
        )

    _axhline_zero()
    plt.xlabel("Step")
    plt.ylabel("Cumulative P&L")
    _finalize_and_save(os.path.join(model_dir, f"cumulative_pnl_best_{criterion}.png"),
                       title=f"Cumulative P&L over Time (best {criterion})",
                       legend_title="Strategy")

def plot_trade_return_distribution(strategies, model_name):
    """
    Distribution of per-trade returns (filter out no-trade steps) for each strategy,
    using its best Sharpe threshold. Colors are FIXED per strategy.
    """
    model_dir = ensure_model_dir(model_name)

    plt.figure(figsize=(12, 6))
    for s in strategies:
        if not s.sharpe_ratios:
            continue
        best_idx = int(np.argmax(s.sharpe_ratios))
        step_pnl = s.all_profit_or_loss[best_idx] if best_idx < len(s.all_profit_or_loss) else []
        if len(step_pnl) == 0:
            continue

        # Heuristic: profit != 0 => a trade was executed
        trade_returns = [p for p in step_pnl if p != 0]
        if len(trade_returns) == 0:
            continue

        sns.kdeplot(trade_returns, label=s.strategy_name, fill=True, alpha=0.45,
                    color=_color_for(s.strategy_name))

    plt.xlabel("Per-Trade Return")
    plt.ylabel("Density")
    _finalize_and_save(os.path.join(model_dir, "trade_return_distribution.png"),
                       title="Trade Return Distribution (best Sharpe)",
                       legend_title="Strategy")

def plot_hitrate_summary(strategies, model_name, criterion="sharpe"):
    """
    Hit-rate (%) per strategy for its best threshold by 'sharpe' or 'profit'.
    Hit-rate = (TP + TN) / (TP + FP + FN + TN) * 100
    """
    assert criterion in ("sharpe", "profit"), "criterion must be 'sharpe' or 'profit'"
    model_dir = ensure_model_dir(model_name)

    names, hitrates = [], []
    suffix = "best_sharpe" if criterion == "sharpe" else "best_profit"

    for s in strategies:
        if criterion == "sharpe":
            if not s.sharpe_ratios:
                continue
            best_idx = int(np.argmax(s.sharpe_ratios))
        else:
            best_idx = int(np.argmax(s.total_profit_or_loss))

        if best_idx >= len(s.confusion_matrices):
            continue

        cm = s.confusion_matrices[best_idx]
        tp, fp, fn, tn = cm.true_positive, cm.false_positive, cm.false_negative, cm.true_negative
        total = tp + fp + fn + tn
        if total == 0:
            continue

        hitrate = (tp + tn) / total * 100.0
        names.append(s.strategy_name)
        hitrates.append(hitrate)

    if not names:
        print("No strategies with valid confusion matrices to plot hit-rate summary.")
        return

    plt.figure(figsize=(10, 6))
    bar_colors = [_color_for(n) for n in names]
    bars = plt.bar(names, hitrates, color=bar_colors)
    plt.ylabel("Hit-Rate (%)")
    plt.xlabel("Strategy")
    plt.ylim(0, 100)
    for i, v in enumerate(hitrates):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom")

    _finalize_and_save(os.path.join(model_dir, f"hitrate_summary_{suffix}.png"),
                       title=f"Hit-Rate by Strategy ({suffix.replace('_', ' ')})")
