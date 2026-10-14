import numpy as np
import pandas as pd
import pytest

from src.risk import (
    historical_es,
    historical_var,
    parametric_es,
    parametric_var,
    portfolio_returns,
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