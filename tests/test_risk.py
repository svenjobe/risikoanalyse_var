import numpy as np
import pandas as pd
import pytest

from src.risk import (
    historical_es,
    historical_var,
    horizon_returns,
    monte_carlo_returns,
    parametric_es,
    parametric_var,
    portfolio_returns,
    drawdown_series,
    max_drawdown,
    longest_underwater_days
)


def test_historical_var_known_values():
    r = np.arange(-50, 50) / 100.0
    assert historical_var(r, 0.95) == pytest.approx(0.4505, abs=1e-3)


def test_expected_shortfall_is_at_least_var():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, 5000)
    for level in (0.95, 0.99):
        assert historical_es(r, level) >= historical_var(r, level)
        assert parametric_es(r, level) >= parametric_var(r, level)


def test_parametric_var_matches_normal_formula():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.01, 200_000)
    assert parametric_var(r, 0.99) == pytest.approx(0.02326, rel=0.02)
    assert historical_var(r, 0.99) == pytest.approx(parametric_var(r, 0.99), rel=0.03)


def test_portfolio_returns_weights_checked():
    df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.00]})
    result = portfolio_returns(df, {"A": 0.5, "B": 0.5})
    assert result.tolist() == pytest.approx([0.02, 0.01])
    with pytest.raises(ValueError):
        portfolio_returns(df, {"A": 0.6, "B": 0.6})


def test_horizon_returns_compounding():
    r = pd.Series([0.01, 0.02, -0.01])
    h = horizon_returns(r, 2)
    assert h.iloc[0] == pytest.approx(1.01 * 1.02 - 1.0)
    assert h.iloc[1] == pytest.approx(1.02 * 0.99 - 1.0)

def test_monte_carlo_var_close_to_parametric_for_one_day():
    rng = np.random.default_rng(4)
    cov = np.array([[1.0, 0.3], [0.3, 1.5]]) * 1e-4
    data = rng.multivariate_normal([0.0003, 0.0002], cov, size=4000)
    df = pd.DataFrame(data, columns=["A", "B"])
    weights = {"A": 0.6, "B": 0.4}
    port = portfolio_returns(df, weights)
    sims = monte_carlo_returns(df, weights, horizon=1, n_sims=100_000, seed=0)
    assert historical_var(sims, 0.95) == pytest.approx(parametric_var(port, 0.95), rel=0.08)

def test_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20])
    assert max_drawdown(r) == pytest.approx(-0.5)
    dd = drawdown_series(r)
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[2] == pytest.approx(0.66 / 1.1 - 1.0)


def test_longest_underwater_days():
    r = pd.Series([0.10, -0.05, 0.01, 0.01, 0.10])
    assert longest_underwater_days(r) == 3