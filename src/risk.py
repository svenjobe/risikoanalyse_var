from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    w = pd.Series(weights).reindex(returns.columns)
    if w.isna().any():
        raise ValueError("Für nicht alle Spalten sind Gewichte vorhanden.")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"Gewichte summieren sich auf {w.sum():.4f}, erwartet 1.0")
    return returns.mul(w, axis=1).sum(axis=1)


def historical_var(returns: pd.Series | np.ndarray, level: float) -> float:
    return float(-np.quantile(np.asarray(returns), 1.0 - level))


def historical_es(returns: pd.Series | np.ndarray, level: float) -> float:
    arr = np.asarray(returns)
    threshold = np.quantile(arr, 1.0 - level)
    tail = arr[arr <= threshold]
    return float(-tail.mean())


def parametric_var(returns: pd.Series | np.ndarray, level: float) -> float:
    arr = np.asarray(returns)
    mu, sigma = arr.mean(), arr.std(ddof=1)
    z = norm.ppf(1.0 - level)
    return float(-(mu + z * sigma))


def parametric_es(returns: pd.Series | np.ndarray, level: float) -> float:
    arr = np.asarray(returns)
    mu, sigma = arr.mean(), arr.std(ddof=1)
    z = norm.ppf(1.0 - level)
    return float(-mu + sigma * norm.pdf(z) / (1.0 - level))

def horizon_returns(returns: pd.Series, horizon: int) -> pd.Series:
    log_ret = np.log1p(returns)
    return np.expm1(log_ret.rolling(horizon).sum()).dropna()


def monte_carlo_returns(
    returns: pd.DataFrame,
    weights: dict[str, float],
    horizon: int,
    n_sims: int,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = pd.Series(weights).reindex(returns.columns).to_numpy()
    mu = returns.mean().to_numpy()
    cov = returns.cov().to_numpy()

    draws = rng.multivariate_normal(mu, cov, size=(n_sims, horizon))
    daily_port = draws @ w                                            
    return np.prod(1.0 + daily_port, axis=1) - 1.0