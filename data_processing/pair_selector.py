import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


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
        # OLS regression to estimate hedge ratio: series_a = alpha + beta * series_b
        ols_model = OLS(series_a, add_constant(series_b)).fit()
        beta = float(ols_model.params[1])

        stat, p_value, crit_values = coint(series_a, series_b)
        return {
            "eg_statistic": float(stat),
            "eg_p_value": float(p_value),
            "eg_crit_1pct": float(crit_values[0]),
            "eg_crit_5pct": float(crit_values[1]),
            "eg_crit_10pct": float(crit_values[2]),
            "cointegrated": bool(p_value < 0.05),
            "hedge_ratio_beta": beta,
        }

    def compute_half_life(self, spread):
        """Fits AR(1) on the spread to estimate mean-reversion half-life in bars."""
        spread = np.array(spread)
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        ols = OLS(spread_diff, spread_lag).fit()
        lambda_coeff = float(ols.params[0])
        if lambda_coeff >= 0:
            return {"half_life_bars": None, "ar1_lambda": lambda_coeff}
        half_life = float(-np.log(2) / lambda_coeff)
        return {"half_life_bars": half_life, "ar1_lambda": lambda_coeff}

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
        corr = self.pearson_correlation(series_a, series_b)
        eg = self.engle_granger_cointegration(series_a, series_b)

        # Use OLS-estimated beta for spread (statistically correct vs beta=1 assumption)
        beta = eg["hedge_ratio_beta"]
        spread = series_a - beta * series_b

        adf = self.adf_on_spread(spread)
        hl = self.compute_half_life(spread)

        self.results = {"pair_name": pair_name, **corr, **eg, **adf, **hl}
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
        print(f"  Hedge ratio (β)   : {r['hedge_ratio_beta']:.6f}")
        print()
        print(f"ADF on Spread       : stat = {r['adf_statistic']:.4f}  (p = {r['adf_p_value']:.4e})  lags = {r['adf_lags_used']}")
        print(f"  Critical values   : 1% = {r['adf_crit_1pct']:.4f}  5% = {r['adf_crit_5pct']:.4f}  10% = {r['adf_crit_10pct']:.4f}")
        print(f"  Result            : {'Spread IS stationary (mean-reversion justified)' if r['spread_stationary'] else 'Spread is NOT stationary'}")
        print()
        hl = r.get("half_life_bars")
        if hl is not None:
            mins = hl * 4
            print(f"Mean-Reversion      : half-life = {hl:.1f} bars (~{mins:.0f} min)  (λ = {r['ar1_lambda']:.6f})")
        else:
            print(f"Mean-Reversion      : not detected (λ = {r['ar1_lambda']:.6f} >= 0, spread diverges)")
        print("=" * 60)

    def to_dataframe(self):
        return pd.DataFrame([self.results])
