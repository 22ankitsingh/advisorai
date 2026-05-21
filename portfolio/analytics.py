"""
portfolio/analytics.py
───────────────────────
Pure analytics functions — no UI, no database calls.

All functions accept DataFrames (from mock_data.py or the DB service)
and return DataFrames or dicts. This makes them easy to test and reuse.

Functions:
  - portfolio_summary()       → total value, cost, P&L, return %
  - allocation_by_class()     → market value & weight per asset class
  - allocation_by_sector()    → market value & weight per sector
  - top_holdings()            → top N positions by market value
  - gain_loss_breakdown()     → sorted P&L per holding
  - performance_metrics()     → return, vol, Sharpe, max drawdown from NAV history
  - drift_analysis()          → actual vs. target allocation deltas
  - rolling_returns()         → 4W / 13W / 26W / 52W returns from NAV
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.mock_data import RISK_FREE_RATE, ASSET_CLASS_METADATA
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-level summary
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_summary(holdings: pd.DataFrame) -> dict:
    """
    Compute high-level portfolio metrics from a holdings DataFrame.

    Args:
        holdings: Output of mock_data.get_holdings_df() — must have
                  columns: market_value, cost_basis, gain_loss.

    Returns:
        Dict with keys:
          total_value, total_cost, total_gain_loss, total_gain_pct,
          num_positions, num_asset_classes, largest_position_weight
    """
    total_value  = holdings["market_value"].sum()
    total_cost   = holdings["cost_basis"].sum()
    gain_loss    = total_value - total_cost
    gain_pct     = (gain_loss / total_cost * 100) if total_cost else 0.0

    # Exclude CASH from position count
    non_cash = holdings[holdings["ticker"] != "CASH"]

    return {
        "total_value":            round(total_value, 2),
        "total_cost":             round(total_cost, 2),
        "total_gain_loss":        round(gain_loss, 2),
        "total_gain_pct":         round(gain_pct, 2),
        "num_positions":          len(non_cash),
        "num_asset_classes":      holdings["asset_class"].nunique(),
        "largest_position_weight": round(holdings["weight"].max(), 2),
        "cash_weight":            round(
            holdings.loc[holdings["ticker"] == "CASH", "weight"].sum(), 2
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Allocation breakdowns
# ─────────────────────────────────────────────────────────────────────────────

def allocation_by_class(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate portfolio by asset class.

    Returns:
        DataFrame with columns:
          asset_class, market_value, weight, gain_loss, count
        Sorted by market_value descending.
    """
    grouped = (
        holdings.groupby("asset_class")
        .agg(
            market_value=("market_value", "sum"),
            cost_basis=("cost_basis", "sum"),
            gain_loss=("gain_loss", "sum"),
            count=("ticker", "count"),
        )
        .reset_index()
    )

    total = grouped["market_value"].sum()
    grouped["weight"] = (grouped["market_value"] / total * 100).round(2)
    grouped["gain_pct"] = (
        (grouped["gain_loss"] / grouped["cost_basis"] * 100)
        .replace([np.inf, -np.inf], 0)
        .round(2)
    )
    grouped["market_value"] = grouped["market_value"].round(2)
    grouped["gain_loss"]    = grouped["gain_loss"].round(2)

    return grouped.sort_values("market_value", ascending=False).reset_index(drop=True)


def allocation_by_sector(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate portfolio by sector (non-null sectors only).

    Returns:
        DataFrame with columns:
          sector, market_value, weight, gain_loss, count
        Sorted by market_value descending.
    """
    sector_df = holdings[holdings["sector"].notna()].copy()

    if sector_df.empty:
        return pd.DataFrame(
            columns=["sector", "market_value", "weight", "gain_loss", "count"]
        )

    grouped = (
        sector_df.groupby("sector")
        .agg(
            market_value=("market_value", "sum"),
            cost_basis=("cost_basis", "sum"),
            gain_loss=("gain_loss", "sum"),
            count=("ticker", "count"),
        )
        .reset_index()
    )

    # Weight as % of total portfolio (including assets without sector)
    total = holdings["market_value"].sum()
    grouped["weight"] = (grouped["market_value"] / total * 100).round(2)
    grouped["gain_pct"] = (
        (grouped["gain_loss"] / grouped["cost_basis"] * 100)
        .replace([np.inf, -np.inf], 0)
        .round(2)
    )
    grouped["market_value"] = grouped["market_value"].round(2)
    grouped["gain_loss"]    = grouped["gain_loss"].round(2)

    return grouped.sort_values("market_value", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Holdings detail
# ─────────────────────────────────────────────────────────────────────────────

def top_holdings(holdings: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return the top N holdings by market value.

    Args:
        holdings: Holdings DataFrame from mock_data.get_holdings_df().
        n:        How many top positions to return.

    Returns:
        DataFrame with columns:
          ticker, name, asset_class, sector, market_value, weight,
          gain_loss, gain_pct
        Sorted by market_value descending.
    """
    cols = ["ticker", "name", "asset_class", "sector",
            "market_value", "weight", "gain_loss", "gain_pct", "price", "avg_cost"]

    available = [c for c in cols if c in holdings.columns]
    return (
        holdings[available]
        .sort_values("market_value", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def gain_loss_breakdown(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Return all holdings sorted by absolute gain/loss — best to worst.

    Useful for "winners and losers" panels.
    """
    non_cash = holdings[holdings["ticker"] != "CASH"].copy()
    return non_cash.sort_values("gain_pct", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics (from NAV history)
# ─────────────────────────────────────────────────────────────────────────────

def performance_metrics(nav_df: pd.DataFrame) -> dict:
    """
    Calculate key performance metrics from a weekly NAV series.

    Args:
        nav_df: Output of mock_data.get_nav_history(). Must have
                columns: date, nav, weekly_return.

    Returns:
        Dict with:
          total_return_pct    — full-period % return
          annualised_return   — CAGR % over the period
          volatility_annual   — annualised standard deviation of weekly returns
          sharpe_ratio        — (annualised_return - risk_free) / volatility
          max_drawdown_pct    — worst peak-to-trough decline %
          best_week_pct       — highest single-week return
          worst_week_pct      — lowest single-week return
          weeks               — number of data points
    """
    if nav_df.empty or len(nav_df) < 2:
        return {}

    nav     = nav_df["nav"].values
    wr      = nav_df["weekly_return"].values / 100   # decimal

    # Total return
    total_return = (nav[-1] / nav[0] - 1) * 100

    # Annualised return (CAGR)
    n_weeks = len(nav)
    years   = n_weeks / 52
    cagr    = ((nav[-1] / nav[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Annualised volatility (weekly std × √52)
    vol_weekly = float(np.std(wr, ddof=1))
    vol_annual = vol_weekly * np.sqrt(52) * 100

    # Sharpe ratio
    rf_weekly  = RISK_FREE_RATE / 52
    excess_wr  = wr - rf_weekly
    sharpe     = (
        float(np.mean(excess_wr) / np.std(excess_wr, ddof=1)) * np.sqrt(52)
        if np.std(excess_wr, ddof=1) > 0 else 0
    )

    # Maximum drawdown
    peak  = np.maximum.accumulate(nav)
    dd    = (nav - peak) / peak * 100
    max_dd = float(np.min(dd))

    return {
        "total_return_pct":  round(total_return, 2),
        "annualised_return": round(cagr, 2),
        "volatility_annual": round(vol_annual, 2),
        "sharpe_ratio":      round(sharpe, 2),
        "max_drawdown_pct":  round(max_dd, 2),
        "best_week_pct":     round(float(np.max(wr)) * 100, 2),
        "worst_week_pct":    round(float(np.min(wr)) * 100, 2),
        "weeks":             n_weeks,
    }


def rolling_returns(nav_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute point-in-time rolling returns: 4W, 13W, 26W, 52W.

    Returns a dict keyed by period label → % return (or None if insufficient data).
    """
    nav = nav_df["nav"].values
    n   = len(nav)
    out = {}

    for label, weeks in [("4W", 4), ("13W", 13), ("26W", 26), ("52W", 52)]:
        if n >= weeks:
            ret = (nav[-1] / nav[-weeks] - 1) * 100
            out[label] = round(ret, 2)
        else:
            out[label] = None

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Drift analysis — actual vs. target allocation
# ─────────────────────────────────────────────────────────────────────────────

def drift_analysis(holdings: pd.DataFrame, target_allocation: dict[str, float]) -> pd.DataFrame:
    """
    Compare actual asset-class weights against the client's target allocation.

    Args:
        holdings:          Holdings DataFrame.
        target_allocation: Dict of asset_class → target weight % (from CLIENTS).

    Returns:
        DataFrame with columns:
          asset_class, actual_weight, target_weight, drift, status
        where status is "Overweight" | "Underweight" | "On Target"
    """
    actual = (
        holdings.groupby("asset_class")["market_value"]
        .sum()
        .div(holdings["market_value"].sum())
        .mul(100)
        .round(2)
    )

    # Build unified DataFrame covering all asset classes in either dict
    all_classes = set(target_allocation.keys()) | set(actual.index)

    rows = []
    for cls in sorted(all_classes):
        act = round(float(actual.get(cls, 0)), 2)
        tgt = float(target_allocation.get(cls, 0))
        drift = round(act - tgt, 2)

        if abs(drift) <= 2.0:
            status = "On Target"
        elif drift > 0:
            status = "Overweight"
        else:
            status = "Underweight"

        rows.append({
            "asset_class":    cls,
            "actual_weight":  act,
            "target_weight":  tgt,
            "drift":          drift,
            "status":         status,
        })

    return pd.DataFrame(rows).sort_values("drift", key=abs, ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Expected portfolio return (weighted)
# ─────────────────────────────────────────────────────────────────────────────

def expected_portfolio_return(holdings: pd.DataFrame) -> float:
    """
    Compute the weighted expected annual return of the portfolio.

    Uses asset-class level expected returns from ASSET_CLASS_METADATA.
    Returns % (e.g. 9.4 means 9.4%).
    """
    total = holdings["market_value"].sum()
    if total == 0:
        return 0.0

    weighted = sum(
        row["market_value"] / total * ASSET_CLASS_METADATA[row["asset_class"]]["expected_return"]
        for _, row in holdings.iterrows()
    )
    return round(weighted * 100, 2)
