from __future__ import annotations

import os

import pandas as pd

from src import config
from src.data_loader import compute_returns, download_adjusted_close
from src.risk import (
    drawdown_series,
    historical_es,
    historical_var,
    kupiec_pof,
    longest_underwater_days,
    max_drawdown,
    monte_carlo_returns,
    portfolio_returns,
    risk_summary,
    rolling_var_backtest,
)
from src.visualize import (
    plot_backtest,
    plot_cumulative,
    plot_drawdown,
    plot_mc_distribution,
    plot_return_histogram,
)


def main() -> None:
    out = config.OUTPUT_DIR
    os.makedirs(out, exist_ok=True)

    tickers = list(config.WEIGHTS)
    print(f"Lade Kurse für {tickers} ({config.START_DATE} bis {config.END_DATE}) ...")
    prices = download_adjusted_close(tickers, config.START_DATE, config.END_DATE)
    returns = compute_returns(prices)
    print(f"{len(returns)} Handelstage, {returns.index[0].date()} bis {returns.index[-1].date()}")

    port = portfolio_returns(returns, config.WEIGHTS)

    # --- VaR / ES -----------------------------------------------------------
    summary = risk_summary(
        returns, config.WEIGHTS, config.CONFIDENCE_LEVELS,
        config.HORIZON_DAYS, config.N_SIMULATIONS, config.RANDOM_SEED,
    )
    pct = summary.copy()
    for col in pct.columns[2:]:
        pct[col] = (pct[col] * 100).round(2)
    pct["Konfidenz"] = (pct["Konfidenz"] * 100).astype(int).astype(str) + " %"
    print("\nVaR / Expected Shortfall (Verlust in % des Portfoliowerts):")
    print(pct.to_string(index=False))
    pct.to_csv(os.path.join(out, "risk_summary.csv"), index=False)

    # --- Drawdown -----------------------------------------------------------
    dd = drawdown_series(port)
    print(f"\nMaximaler Drawdown: {max_drawdown(port) * 100:.1f} %")
    print(f"Längste Phase unter Höchststand: {longest_underwater_days(port)} Handelstage")

    # --- Backtest -----------------------------------------------------------
    rows = []
    backtests = {}
    for level in config.CONFIDENCE_LEVELS:
        bt = rolling_var_backtest(port, level, config.ROLLING_WINDOW)
        backtests[level] = bt
        n_exc, n_obs = int(bt["exception"].sum()), len(bt)
        lr, p_value = kupiec_pof(n_exc, n_obs, 1.0 - level)
        rows.append({
            "Konfidenz": f"{level:.0%}",
            "Beobachtungen": n_obs,
            "Ausnahmen": n_exc,
            "Erwartet": round(n_obs * (1.0 - level), 1),
            "Kupiec LR": round(lr, 2),
            "p-Wert": round(p_value, 3),
        })
    bt_table = pd.DataFrame(rows)
    print("\nBacktest des rollierenden historischen VaR (Kupiec-Test):")
    print(bt_table.to_string(index=False))
    bt_table.to_csv(os.path.join(out, "backtest_summary.csv"), index=False)

    # --- Grafiken -----------------------------------------------------------
    plot_cumulative(port, os.path.join(out, "wertentwicklung.png"))
    plot_return_histogram(
        port,
        historical_var(port, 0.95), historical_var(port, 0.99), historical_es(port, 0.99),
        os.path.join(out, "renditeverteilung.png"),
    )
    plot_drawdown(port, dd, os.path.join(out, "drawdown.png"))
    plot_backtest(backtests[0.99], 0.99, os.path.join(out, "var_backtest_99.png"))

    sims = monte_carlo_returns(
        returns, config.WEIGHTS, config.HORIZON_DAYS, config.N_SIMULATIONS, config.RANDOM_SEED
    )
    plot_mc_distribution(
        sims, historical_var(sims, 0.99), historical_es(sims, 0.99),
        config.HORIZON_DAYS, 0.99, os.path.join(out, "monte_carlo.png"),
    )
    print(f"\nFertig. Ergebnisse in '{out}/'.")


if __name__ == "__main__":
    main()