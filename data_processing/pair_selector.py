import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import coint, adfuller


class PairSelector:
    def __init__(self):
        self.results = {}

    def pearson_correlation(self, series_a, series_b):
        r, p_value = pearsonr(series_a, series_b)
        return {
            "pearson_r": float(r),
            "pearson_p_value": float(p_value),
        }

    def engle_granger_cointegration(self, series_a, series_b):
        stat, p_value, crit_values = coint(series_a, series_b)
        return {
            "eg_statistic": float(stat),
            "eg_p_value": float(p_value),
            "eg_crit_1pct": float(crit_values[0]),
            "eg_crit_5pct": float(crit_values[1]),
            "eg_crit_10pct": float(crit_values[2]),
            "cointegrated": bool(p_value < 0.05),
        }

    def adf_on_spread(self, spread):
        result = adfuller(spread, autolag="AIC")
        return {
            "adf_statistic": float(result[0]),
            "adf_p_value": float(result[1]),
            "adf_lags_used": int(result[2]),
            "adf_crit_1pct": float(result[4]["1%"]),
            "adf_crit_5pct": float(result[4]["5%"]),
            "adf_crit_10pct": float(result[4]["10%"]),
            "spread_stationary": bool(result[1] < 0.05),
        }

    def run_full_analysis(self, series_a, series_b, pair_name="Pair"):
        spread = series_a - series_b

        corr = self.pearson_correlation(series_a, series_b)
        eg = self.engle_granger_cointegration(series_a, series_b)
        adf = self.adf_on_spread(spread)

        self.results = {"pair_name": pair_name, **corr, **eg, **adf}
        self._print_report()
        return self.results

    def _print_report(self):
        r = self.results
        print("=" * 60)
        print(f"PAIR ANALYSIS REPORT -- {r['pair_name']}")
        print("=" * 60)
        print(f"Pearson Correlation : r = {r['pearson_r']:.4f}  (p = {r['pearson_p_value']:.4e})")
        print()
        print(f"Engle-Granger Test  : stat = {r['eg_statistic']:.4f}  (p = {r['eg_p_value']:.4e})")
        print(f"  Critical values   : 1% = {r['eg_crit_1pct']:.4f}  5% = {r['eg_crit_5pct']:.4f}  10% = {r['eg_crit_10pct']:.4f}")
        print(f"  Result            : {'COINTEGRATED (p < 0.05)' if r['cointegrated'] else 'NOT cointegrated (p >= 0.05)'}")
        print()
        print(f"ADF on Spread       : stat = {r['adf_statistic']:.4f}  (p = {r['adf_p_value']:.4e})  lags = {r['adf_lags_used']}")
        print(f"  Critical values   : 1% = {r['adf_crit_1pct']:.4f}  5% = {r['adf_crit_5pct']:.4f}  10% = {r['adf_crit_10pct']:.4f}")
        print(f"  Result            : {'Spread IS stationary (mean-reversion justified)' if r['spread_stationary'] else 'Spread is NOT stationary'}")
        print("=" * 60)

    def to_dataframe(self):
        return pd.DataFrame([self.results])
