import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def _ensure_pair_analysis_dir(data_file_path: str) -> str:
    dataset_name = os.path.splitext(os.path.basename(data_file_path))[0]
    dir_path = os.path.join("plot", dataset_name, "pair_analysis")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def plot_rolling_correlation(series_a, series_b, data_file_path: str, window: int = 200):
    """Rolling Pearson correlation between the two price series."""
    dir_path = _ensure_pair_analysis_dir(data_file_path)

    s_a = pd.Series(series_a.astype(float))
    s_b = pd.Series(series_b.astype(float))
    rolling_corr = s_a.rolling(window).corr(s_b)

    plt.figure(figsize=(12, 5))
    plt.plot(rolling_corr.values, color="#0072B2", linewidth=1.8, label=f"Rolling corr (window={window})")
    plt.axhline(0, color="#888888", linewidth=1, linestyle="--")
    plt.axhline(0.8, color="#E69F00", linewidth=1, linestyle=":", label="r = 0.8")
    plt.xlabel("Observation")
    plt.ylabel("Pearson r")
    plt.title("Rolling Correlation Between Price Series")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(dir_path, "rolling_correlation.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {save_path}")


def plot_spread_with_bands(spread, data_file_path: str):
    """Raw spread (price_A - price_B) with ±1σ and ±2σ bands."""
    dir_path = _ensure_pair_analysis_dir(data_file_path)

    spread = np.array(spread, dtype=float)
    mean = np.mean(spread)
    std = np.std(spread)

    plt.figure(figsize=(12, 5))
    plt.plot(spread, color="#0072B2", linewidth=1.2, label="Spread", alpha=0.85)
    plt.axhline(mean, color="#000000", linewidth=1.5, linestyle="-", label="Mean")
    plt.axhline(mean + std, color="#E69F00", linewidth=1.2, linestyle="--", label="+1σ")
    plt.axhline(mean - std, color="#E69F00", linewidth=1.2, linestyle="--", label="-1σ")
    plt.axhline(mean + 2 * std, color="#D55E00", linewidth=1.2, linestyle=":", label="+2σ")
    plt.axhline(mean - 2 * std, color="#D55E00", linewidth=1.2, linestyle=":", label="-2σ")
    plt.fill_between(range(len(spread)), mean - std, mean + std, alpha=0.10, color="#E69F00")
    plt.fill_between(range(len(spread)), mean - 2 * std, mean + 2 * std, alpha=0.05, color="#D55E00")
    plt.xlabel("Observation")
    plt.ylabel("Spread (A - B)")
    plt.title("Spread with Mean-Reversion Bands")
    plt.legend(fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(dir_path, "spread_with_bands.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {save_path}")


def save_pair_analysis_csv(results: dict, data_file_path: str):
    """Saves the analysis results dict to CSV inside the pair_analysis folder."""
    dir_path = _ensure_pair_analysis_dir(data_file_path)
    df = pd.DataFrame([results])
    save_path = os.path.join(dir_path, "pair_analysis_results.csv")
    df.to_csv(save_path, index=False)
    print(f"Pair analysis CSV saved to: {save_path}")
    return save_path
