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

def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def longest_underwater_days(returns: pd.Series) -> int:
    dd = drawdown_series(returns)
    underwater = dd < 0
    if not underwater.any():
        return 0
    groups = (~underwater).cumsum()
    return int(underwater.groupby(groups).sum().max())

def rolling_var_backtest(returns: pd.Series, level: float, window: int) -> pd.DataFrame:
    threshold = returns.rolling(window).quantile(1.0 - level).shift(1)
    out = pd.DataFrame({"return": returns, "var": -threshold}).dropna()
    out["exception"] = out["return"] < -out["var"]
    return out


def kupiec_pof(n_exceptions: int, n_obs: int, p: float) -> tuple[float, float]:
    x, n = n_exceptions, n_obs
    if n <= 0:
        raise ValueError("n_obs muss positiv sein.")
    pi_hat = x / n

    def log_lik(prob: float) -> float:
        term_ok = (n - x) * np.log(1.0 - prob) if n - x > 0 else 0.0
        term_exc = x * np.log(prob) if x > 0 else 0.0
        return term_ok + term_exc

    lr = -2.0 * (log_lik(p) - log_lik(pi_hat))
    return float(lr), float(chi2.sf(lr, df=1))

def risk_summary(
    returns: pd.DataFrame,
    weights: dict[str, float],
    levels: tuple[float, ...],
    horizon: int,
    n_sims: int,
    seed: int,
) -> pd.DataFrame:
    port = portfolio_returns(returns, weights)
    port_h = horizon_returns(port, horizon)
    sims = monte_carlo_returns(returns, weights, horizon, n_sims, seed)
    sims_1d = monte_carlo_returns(returns, weights, 1, n_sims, seed)

    rows = []
    for level in levels:
        rows.append({
            "Methode": "Historisch",
            "Konfidenz": level,
            "VaR 1 Tag": historical_var(port, level),
            "ES 1 Tag": historical_es(port, level),
            f"VaR {horizon} Tage": historical_var(port_h, level),
        })
        rows.append({
            "Methode": "Parametrisch (Normal)",
            "Konfidenz": level,
            "VaR 1 Tag": parametric_var(port, level),
            "ES 1 Tag": parametric_es(port, level),
            f"VaR {horizon} Tage": parametric_var(port_h, level),
        })
        rows.append({
            "Methode": "Monte-Carlo",
            "Konfidenz": level,
            "VaR 1 Tag": historical_var(sims_1d, level),
            "ES 1 Tag": historical_es(sims_1d, level),
            f"VaR {horizon} Tage": historical_var(sims, level),
        })
    return pd.DataFrame(rows)