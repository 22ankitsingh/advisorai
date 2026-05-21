"""
portfolio/mock_data.py
───────────────────────
Self-contained mock financial dataset for portfolio analytics.

Provides:
  - Client profiles (same 5 from seed.py, but enriched)
  - Detailed holdings with asset metadata
  - 12-month historical NAV series per client (weekly, simulated)
  - Benchmark (S&P 500 proxy) history for comparison
  - Asset-class level volatility/beta metadata

Design rule:
  All functions return plain Python structures (dicts, lists) or
  pandas DataFrames — no DB calls. The analytics and risk modules
  import from here, keeping them database-agnostic and fast.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Client registry
# ─────────────────────────────────────────────────────────────────────────────

CLIENTS: dict[str, dict] = {
    "sarah_mitchell": {
        "id":           "sarah_mitchell",
        "name":         "Sarah Mitchell",
        "email":        "sarah.mitchell@example.com",
        "risk_profile": "aggressive",
        "target_allocation": {           # Target % by asset class
            "equity":      70,
            "etf":         15,
            "alternative": 10,
            "bond":         0,
            "cash":         5,
        },
        "benchmark":    "QQQ",           # NASDAQ-100 proxy
        "inception":    date(2022, 1, 1),
        "advisor_notes": "Tech entrepreneur. Interested in growth stocks and early-stage investments.",
    },
    "robert_chen": {
        "id":           "robert_chen",
        "name":         "Robert Chen",
        "email":        "robert.chen@example.com",
        "risk_profile": "moderate",
        "target_allocation": {
            "equity":      40,
            "etf":         35,
            "bond":        15,
            "alternative":  0,
            "cash":        10,
        },
        "benchmark":    "SPY",
        "inception":    date(2020, 6, 1),
        "advisor_notes": "Software engineer, mid-career. Balanced growth and income focus.",
    },
    "eleanor_vasquez": {
        "id":           "eleanor_vasquez",
        "name":         "Eleanor Vasquez",
        "email":        "eleanor.vasquez@example.com",
        "risk_profile": "conservative",
        "target_allocation": {
            "equity":      20,
            "etf":         25,
            "bond":        45,
            "alternative":  0,
            "cash":        10,
        },
        "benchmark":    "AGG",          # Bond index proxy
        "inception":    date(2019, 3, 1),
        "advisor_notes": "Retired school principal. Capital preservation is top priority.",
    },
    "marcus_thompson": {
        "id":           "marcus_thompson",
        "name":         "Marcus Thompson",
        "email":        "marcus.thompson@example.com",
        "risk_profile": "moderate",
        "target_allocation": {
            "equity":      30,
            "etf":         50,
            "bond":         0,
            "alternative": 10,
            "cash":        10,
        },
        "benchmark":    "SPY",
        "inception":    date(2021, 9, 1),
        "advisor_notes": "Small business owner. Wants to diversify outside of his business.",
    },
    "priya_kapoor": {
        "id":           "priya_kapoor",
        "name":         "Priya Kapoor",
        "email":        "priya.kapoor@example.com",
        "risk_profile": "aggressive",
        "target_allocation": {
            "equity":      65,
            "etf":         25,
            "bond":         0,
            "alternative":  0,
            "cash":        10,
        },
        "benchmark":    "XBI",          # Biotech sector proxy
        "inception":    date(2021, 1, 1),
        "advisor_notes": "Physician. High income, long time horizon, interested in healthcare sector.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Asset metadata — volatility, beta, expected annual return by asset class
# Used by risk_engine.py to compute risk scores without live price feeds.
# ─────────────────────────────────────────────────────────────────────────────

ASSET_CLASS_METADATA: dict[str, dict] = {
    "equity":      {"annual_vol": 0.20, "beta": 1.10, "expected_return": 0.10},
    "etf":         {"annual_vol": 0.15, "beta": 0.95, "expected_return": 0.08},
    "bond":        {"annual_vol": 0.06, "beta": 0.15, "expected_return": 0.04},
    "alternative": {"annual_vol": 0.55, "beta": 1.40, "expected_return": 0.18},
    "cash":        {"annual_vol": 0.00, "beta": 0.00, "expected_return": 0.05},
}

# Risk-free rate for Sharpe ratio calculation
RISK_FREE_RATE: float = 0.053   # ~5.3% (approximate 2024 US T-bill rate)


# ─────────────────────────────────────────────────────────────────────────────
# Holdings data
# ─────────────────────────────────────────────────────────────────────────────

_RAW_HOLDINGS: dict[str, list[dict]] = {
    "sarah_mitchell": [
        {"ticker": "NVDA", "name": "NVIDIA Corporation",     "asset_class": "equity",      "sector": "Technology",              "qty": 120,  "avg_cost": 410.00, "price": 875.50},
        {"ticker": "TSLA", "name": "Tesla Inc.",             "asset_class": "equity",      "sector": "Consumer Discretionary",  "qty": 80,   "avg_cost": 180.00, "price": 245.30},
        {"ticker": "META", "name": "Meta Platforms",         "asset_class": "equity",      "sector": "Technology",              "qty": 150,  "avg_cost": 290.00, "price": 480.20},
        {"ticker": "ARKK", "name": "ARK Innovation ETF",     "asset_class": "etf",         "sector": None,                      "qty": 500,  "avg_cost":  45.00, "price":  52.10},
        {"ticker": "BTC",  "name": "Bitcoin (via ETF)",      "asset_class": "alternative", "sector": None,                      "qty": 3,    "avg_cost": 28000., "price": 62000.0},
        {"ticker": "CASH", "name": "Cash & Equivalents",     "asset_class": "cash",        "sector": None,                      "qty": 1,    "avg_cost": 50000., "price": 50000.0},
    ],
    "robert_chen": [
        {"ticker": "AAPL", "name": "Apple Inc.",             "asset_class": "equity",      "sector": "Technology",              "qty": 100,  "avg_cost": 155.00, "price": 189.40},
        {"ticker": "VTI",  "name": "Vanguard Total Mkt ETF", "asset_class": "etf",         "sector": None,                      "qty": 200,  "avg_cost": 195.00, "price": 238.70},
        {"ticker": "BND",  "name": "Vanguard Bond ETF",      "asset_class": "bond",        "sector": None,                      "qty": 300,  "avg_cost":  72.00, "price":  74.20},
        {"ticker": "MSFT", "name": "Microsoft Corporation",  "asset_class": "equity",      "sector": "Technology",              "qty": 60,   "avg_cost": 320.00, "price": 415.80},
        {"ticker": "VNQ",  "name": "Vanguard Real Estate",   "asset_class": "etf",         "sector": "Real Estate",             "qty": 150,  "avg_cost":  82.00, "price":  88.90},
        {"ticker": "CASH", "name": "Cash & Equivalents",     "asset_class": "cash",        "sector": None,                      "qty": 1,    "avg_cost": 30000., "price": 30000.0},
    ],
    "eleanor_vasquez": [
        {"ticker": "TLT",  "name": "iShares 20Y Treasury",   "asset_class": "bond",        "sector": None,                      "qty": 400,  "avg_cost":  95.00, "price":  91.30},
        {"ticker": "VYM",  "name": "Vanguard High Div ETF",  "asset_class": "etf",         "sector": None,                      "qty": 500,  "avg_cost": 100.00, "price": 112.40},
        {"ticker": "JNJ",  "name": "Johnson & Johnson",      "asset_class": "equity",      "sector": "Healthcare",              "qty": 200,  "avg_cost": 155.00, "price": 148.90},
        {"ticker": "PG",   "name": "Procter & Gamble",       "asset_class": "equity",      "sector": "Consumer Staples",        "qty": 150,  "avg_cost": 140.00, "price": 158.20},
        {"ticker": "AGG",  "name": "iShares Core Bond ETF",  "asset_class": "bond",        "sector": None,                      "qty": 600,  "avg_cost":  98.00, "price":  96.80},
        {"ticker": "CASH", "name": "Cash & Equivalents",     "asset_class": "cash",        "sector": None,                      "qty": 1,    "avg_cost": 250000.,"price": 250000.},
    ],
    "marcus_thompson": [
        {"ticker": "SPY",  "name": "SPDR S&P 500 ETF",       "asset_class": "etf",         "sector": None,                      "qty": 100,  "avg_cost": 420.00, "price": 528.60},
        {"ticker": "QQQ",  "name": "Invesco NASDAQ-100 ETF", "asset_class": "etf",         "sector": None,                      "qty": 50,   "avg_cost": 340.00, "price": 440.20},
        {"ticker": "AMZN", "name": "Amazon.com Inc.",         "asset_class": "equity",      "sector": "Consumer Discretionary",  "qty": 40,   "avg_cost": 130.00, "price": 185.70},
        {"ticker": "GLD",  "name": "SPDR Gold Shares",       "asset_class": "etf",         "sector": None,                      "qty": 80,   "avg_cost": 170.00, "price": 210.40},
        {"ticker": "CASH", "name": "Cash & Equivalents",     "asset_class": "cash",        "sector": None,                      "qty": 1,    "avg_cost": 20000., "price": 20000.0},
    ],
    "priya_kapoor": [
        {"ticker": "UNH",  "name": "UnitedHealth Group",     "asset_class": "equity",      "sector": "Healthcare",              "qty": 80,   "avg_cost": 480.00, "price": 520.30},
        {"ticker": "ABBV", "name": "AbbVie Inc.",            "asset_class": "equity",      "sector": "Healthcare",              "qty": 120,  "avg_cost": 140.00, "price": 175.80},
        {"ticker": "GOOGL","name": "Alphabet Inc.",          "asset_class": "equity",      "sector": "Technology",              "qty": 60,   "avg_cost": 130.00, "price": 175.40},
        {"ticker": "XBI",  "name": "SPDR Biotech ETF",       "asset_class": "etf",         "sector": "Healthcare",              "qty": 200,  "avg_cost":  75.00, "price":  88.20},
        {"ticker": "MSFT", "name": "Microsoft Corporation",  "asset_class": "equity",      "sector": "Technology",              "qty": 50,   "avg_cost": 380.00, "price": 415.80},
        {"ticker": "CASH", "name": "Cash & Equivalents",     "asset_class": "cash",        "sector": None,                      "qty": 1,    "avg_cost": 40000., "price": 40000.0},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API — Holdings
# ─────────────────────────────────────────────────────────────────────────────

def get_holdings_df(client_id: str) -> pd.DataFrame:
    """
    Return all holdings for a client as a DataFrame with computed columns.

    Computed columns added:
      market_value, cost_basis, gain_loss, gain_pct, weight

    Args:
        client_id: One of the keys in CLIENTS.

    Returns:
        DataFrame with one row per holding.

    Raises:
        KeyError if client_id is not found.
    """
    if client_id not in _RAW_HOLDINGS:
        raise KeyError(f"Unknown client_id: {client_id!r}. Valid: {list(_RAW_HOLDINGS)}")

    df = pd.DataFrame(_RAW_HOLDINGS[client_id])

    # Computed metrics
    df["market_value"] = df["qty"] * df["price"]
    df["cost_basis"]   = df["qty"] * df["avg_cost"]
    df["gain_loss"]    = df["market_value"] - df["cost_basis"]
    df["gain_pct"]     = ((df["price"] - df["avg_cost"]) / df["avg_cost"]) * 100
    df["weight"]       = (df["market_value"] / df["market_value"].sum()) * 100

    # Join asset class metadata
    df["annual_vol"]       = df["asset_class"].map(lambda c: ASSET_CLASS_METADATA[c]["annual_vol"])
    df["beta"]             = df["asset_class"].map(lambda c: ASSET_CLASS_METADATA[c]["beta"])
    df["expected_return"]  = df["asset_class"].map(lambda c: ASSET_CLASS_METADATA[c]["expected_return"])

    return df.reset_index(drop=True)


def get_all_clients() -> list[dict]:
    """Return list of all client profile dicts (without holdings)."""
    return list(CLIENTS.values())


def get_client(client_id: str) -> dict:
    """Return a single client profile dict."""
    return CLIENTS[client_id]


# ─────────────────────────────────────────────────────────────────────────────
# Historical NAV simulation
# ─────────────────────────────────────────────────────────────────────────────

# Annual return & daily volatility parameters per risk profile
_NAV_PARAMS: dict[str, dict] = {
    "aggressive":   {"annual_return": 0.188, "annual_vol": 0.24},
    "moderate":     {"annual_return": 0.112, "annual_vol": 0.13},
    "conservative": {"annual_return": 0.048, "annual_vol": 0.06},
}

# Benchmark parameters (S&P 500 proxy)
_BENCHMARK_PARAMS = {"annual_return": 0.142, "annual_vol": 0.155}


def get_nav_history(
    client_id: str,
    weeks: int = 52,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate simulated weekly NAV (Net Asset Value) history.

    Uses Geometric Brownian Motion with parameters tuned to the client's
    risk profile. The starting NAV is the client's current portfolio value.

    Args:
        client_id: Client identifier.
        weeks:     Number of weekly data points to generate.
        seed:      Optional random seed for reproducibility.

    Returns:
        DataFrame with columns: date, nav, weekly_return, cumulative_return
    """
    if seed is None:
        # Deterministic per client — same data every render
        seed = abs(hash(client_id)) % 100_000

    rng = np.random.default_rng(seed)

    profile = CLIENTS[client_id]["risk_profile"]
    params  = _NAV_PARAMS[profile]

    # Starting NAV = current portfolio value
    df_holdings   = get_holdings_df(client_id)
    start_nav     = df_holdings["market_value"].sum()

    # Scale GBM parameters to weekly
    dt            = 1 / 52
    mu            = params["annual_return"]
    sigma         = params["annual_vol"]

    # Simulate weekly log-returns: r_t = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
    log_returns   = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(weeks)
    nav_series    = start_nav * np.exp(np.cumsum(log_returns))

    # Build date range (weekly, ending today)
    end_date   = date.today()
    dates      = [end_date - timedelta(weeks=weeks - i) for i in range(weeks)]

    df = pd.DataFrame({
        "date":             pd.to_datetime(dates),
        "nav":              nav_series,
        "log_return":       log_returns,
    })

    # Weekly % return
    df["weekly_return"]     = (np.exp(df["log_return"]) - 1) * 100
    # Cumulative return vs. the inferred start (nav_series[0] / original * 100)
    df["cumulative_return"] = ((df["nav"] / nav_series[0]) - 1) * 100

    return df.drop(columns=["log_return"]).reset_index(drop=True)


def get_benchmark_history(weeks: int = 52, seed: int = 42) -> pd.DataFrame:
    """
    Generate simulated S&P 500 benchmark NAV history for comparison.

    Returns the same column format as get_nav_history() but starting at 100
    (index-normalised) so it overlays cleanly on any client's chart.

    Args:
        weeks: Number of weekly data points.
        seed:  Fixed seed for a consistent benchmark.

    Returns:
        DataFrame with columns: date, nav, weekly_return, cumulative_return
    """
    rng = np.random.default_rng(seed)
    mu, sigma = _BENCHMARK_PARAMS["annual_return"], _BENCHMARK_PARAMS["annual_vol"]
    dt        = 1 / 52

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(weeks)
    nav_series  = 100.0 * np.exp(np.cumsum(log_returns))

    end_date = date.today()
    dates    = [end_date - timedelta(weeks=weeks - i) for i in range(weeks)]

    df = pd.DataFrame({
        "date":         pd.to_datetime(dates),
        "nav":          nav_series,
        "log_return":   log_returns,
    })
    df["weekly_return"]     = (np.exp(df["log_return"]) - 1) * 100
    df["cumulative_return"] = ((df["nav"] / nav_series[0]) - 1) * 100

    return df.drop(columns=["log_return"]).reset_index(drop=True)


def get_normalised_nav_history(client_id: str, weeks: int = 52) -> pd.DataFrame:
    """
    Return client NAV history normalised to 100 at inception.
    Useful for overlaying multiple clients or the benchmark on one chart.
    """
    df = get_nav_history(client_id, weeks=weeks)
    start = df["nav"].iloc[0]
    df["nav_indexed"] = (df["nav"] / start) * 100
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Monthly returns heatmap data
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def get_monthly_returns(client_id: str, years: int = 2) -> pd.DataFrame:
    """
    Return a (years × 12) DataFrame of simulated monthly returns (%).

    Rows = years (most recent first), Columns = months.
    Useful for a heatmap chart in the dashboard.
    """
    profile = CLIENTS[client_id]["risk_profile"]
    params  = _NAV_PARAMS[profile]
    mu      = params["annual_return"] / 12
    sigma   = params["annual_vol"] / np.sqrt(12)

    rng     = np.random.default_rng(abs(hash(client_id + "monthly")) % 100_000)
    current_year = date.today().year

    rows = {}
    for y in range(years):
        year = current_year - y
        returns = rng.normal(mu, sigma, 12) * 100
        rows[str(year)] = dict(zip(_MONTH_NAMES, returns))

    df = pd.DataFrame(rows).T   # rows = years, columns = months
    return df[_MONTH_NAMES]     # ensure column order
