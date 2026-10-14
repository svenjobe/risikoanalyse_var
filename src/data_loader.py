from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf

def download_adjusted_close(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(
            "Keine Kursdaten erhalten. Internetverbindung und Ticker pruefen."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.dropna(how="all")
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise ValueError(f"Keine Daten fuer: {missing}")

    close = close[list(tickers)].ffill(limit=3).dropna()
    return close


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")