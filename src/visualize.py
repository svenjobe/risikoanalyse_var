from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting import (
    AQUA, BLUE, GRID, INK_SECONDARY, ORANGE, RED, VIOLET,
    save_figure, set_title, style_axis,
)


def plot_cumulative(port: pd.Series, path: str) -> None:
    """Wertentwicklung von 1 investiertem Dollar."""
    wealth = (1.0 + port).cumprod()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(wealth.index, wealth.values, color=BLUE, linewidth=1.8)
    ax.axhline(1.0, color=GRID, linewidth=1)
    set_title(ax, "Wertentwicklung des Portfolios (Start = 1)")
    ax.set_ylabel("Portfoliowert")
    style_axis(ax)
    save_figure(fig, path)


def plot_return_histogram(port: pd.Series, var95: float, var99: float, es99: float, path: str) -> None:
    """Verteilung der Tagesrenditen mit VaR- und ES-Marken."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(port.values * 100, bins=80, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(-var95 * 100, color=ORANGE, linewidth=2, label=f"VaR 95 %: {var95 * 100:.2f} %")
    ax.axvline(-var99 * 100, color=RED, linewidth=2, label=f"VaR 99 %: {var99 * 100:.2f} %")
    ax.axvline(-es99 * 100, color=VIOLET, linewidth=2, linestyle="--", label=f"ES 99 %: {es99 * 100:.2f} %")
    set_title(ax, "Verteilung der Tagesrenditen (historisch)")
    ax.set_xlabel("Tagesrendite (%)")
    ax.set_ylabel("Anzahl Handelstage")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax)
    save_figure(fig, path)


def plot_drawdown(port: pd.Series, dd: pd.Series, path: str) -> None:
    """Drawdown-Verlauf (Unterwasser-Chart)."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(dd.index, dd.values * 100, 0, color=RED, alpha=0.25)
    ax.plot(dd.index, dd.values * 100, color=RED, linewidth=1.2)
    trough = dd.idxmin()
    ax.annotate(
        f"Max. Drawdown {dd.min() * 100:.1f} %",
        xy=(trough, dd.min() * 100),
        xytext=(10, 12), textcoords="offset points",
        fontsize=9, color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, lw=0.8),
    )
    set_title(ax, "Drawdown: Abstand zum bisherigen Höchststand")
    ax.set_ylabel("Drawdown (%)")
    style_axis(ax)
    save_figure(fig, path)


def plot_backtest(bt: pd.DataFrame, level: float, path: str) -> None:
    """Tagesrenditen gegen den rollierenden VaR; Ausnahmen sind markiert."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(bt.index, bt["return"] * 100, color=BLUE, linewidth=0.7, alpha=0.8, label="Tagesrendite")
    ax.plot(bt.index, -bt["var"] * 100, color=ORANGE, linewidth=1.4, label=f"-VaR {level:.0%} (rollierend)")
    exc = bt[bt["exception"]]
    ax.scatter(exc.index, exc["return"] * 100, color=RED, s=14, zorder=3, label=f"Ausnahmen ({len(exc)})")
    set_title(ax, f"VaR-Backtest {level:.0%}: Tagesrenditen gegen rollierenden VaR")
    ax.set_ylabel("Rendite (%)")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower left")
    style_axis(ax)
    save_figure(fig, path)


def plot_mc_distribution(sims: np.ndarray, var: float, es: float, horizon: int, level: float, path: str) -> None:
    """Verteilung der simulierten Mehrtagesrenditen."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(sims * 100, bins=80, color=AQUA, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(-var * 100, color=ORANGE, linewidth=2, label=f"VaR {level:.0%}: {var * 100:.2f} %")
    ax.axvline(-es * 100, color=VIOLET, linewidth=2, linestyle="--", label=f"ES {level:.0%}: {es * 100:.2f} %")
    set_title(ax, f"Monte-Carlo-Simulation: {horizon}-Tage-Rendite")
    ax.set_xlabel(f"Rendite über {horizon} Handelstage (%)")
    ax.set_ylabel("Anzahl Simulationen")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax)
    save_figure(fig, path)