from __future__ import annotations

from datetime import date, timedelta

# Portfolio aus liquiden ETFs (Yahoo-Finance-Ticker), alle Kurse in USD
WEIGHTS: dict[str, float] = {
    "SPY": 0.40,  # US-Aktien (S&P 500)
    "EEM": 0.15,  # Schwellenländer-Aktien
    "TLT": 0.20,  # US-Staatsanleihen, lange Laufzeit
    "GLD": 0.15,  # Gold
    "VNQ": 0.10,  # US-Immobilien (REITs)
}

# Betrachtungszeitraum: die letzten 10 Jahre
END_DATE: date = date.today()
START_DATE: date = END_DATE - timedelta(days=10 * 365)

# Risikokennzahlen
CONFIDENCE_LEVELS: tuple[float, ...] = (0.95, 0.99)
HORIZON_DAYS: int = 10        # Haltedauer für den Mehrtages-VaR
ROLLING_WINDOW: int = 250     # Fensterlänge (Handelstage) für den Backtest

# Monte-Carlo-Simulation
N_SIMULATIONS: int = 20_000
RANDOM_SEED: int = 42

OUTPUT_DIR: str = "output"