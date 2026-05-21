"""
portfolio/risk_engine.py
─────────────────────────
Risk scoring and analysis engine — pure functions, no UI or DB calls.

Produces:
  - Composite risk score (0–100) with component breakdown
  - Concentration risk via Herfindahl-Hirschman Index (HHI)
  - Volatility-weighted portfolio risk
  - Beta (market sensitivity) estimate
  - Risk-profile alignment score
  - Diversification score
  - Human-readable risk flags for the dashboard

Risk Score composition (0–100 scale, higher = riskier):
  ┌──────────────────────────────┬────────┐
  │ Component                    │ Weight │
  ├──────────────────────────────┼────────┤
  │ Concentration (HHI)          │  30%   │
  │ Asset-class volatility       │  30%   │
  │ Portfolio beta               │  20%   │
  │ Profile alignment            │  20%   │
  └──────────────────────────────┴────────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from portfolio.mock_data import ASSET_CLASS_METADATA, CLIENTS
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Risk band thresholds
# ─────────────────────────────────────────────────────────────────────────────

RISK_BANDS = [
    (0,  25, "Very Low",  "#2ed573"),
    (25, 45, "Low",       "#7bed9f"),
    (45, 60, "Moderate",  "#ffa502"),
    (60, 75, "High",      "#ff6b35"),
    (75, 101,"Very High", "#ff4757"),
]

# Expected risk score range per investor profile (for alignment scoring)
_PROFILE_RISK_RANGE: dict[str, tuple[int, int]] = {
    "conservative": (10, 40),
    "moderate":     (35, 65),
    "aggressive":   (58, 90),
}

# Maximum acceptable single-position weight before triggering concentration flag
MAX_SINGLE_POSITION_PCT = 25.0

# Maximum sector concentration before triggering sector flag
MAX_SECTOR_PCT          = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Output structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskFlag:
    """A single human-readable risk observation."""
    severity:    str   # "critical" | "high" | "medium" | "low" | "info"
    category:   str   # "concentration" | "volatility" | "alignment" | "diversification"
    title:      str
    detail:     str

    @property
    def emoji(self) -> str:
        return {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "info": "ℹ️"}.get(
            self.severity, "⚪"
        )


@dataclass
class RiskReport:
    """Full risk assessment for a client portfolio."""
    client_id:            str
    risk_profile:         str

    # Composite score
    overall_score:        float      # 0–100
    risk_band:            str        # "Very Low" | "Low" | "Moderate" | "High" | "Very High"
    risk_colour:          str        # Hex colour for UI

    # Component scores (0–100 each)
    concentration_score:  float
    volatility_score:     float
    beta_score:           float
    alignment_score:      float

    # Derived metrics
    hhi:                  float      # Herfindahl-Hirschman Index (0–1)
    portfolio_beta:       float      # Weighted portfolio beta
    portfolio_vol:        float      # Annualised portfolio volatility %
    diversification_score: float     # 0–100 (higher = more diversified)

    # Human-readable flags
    flags:                list[RiskFlag] = field(default_factory=list)

    @property
    def is_aligned(self) -> bool:
        """True if the portfolio risk is within the profile's expected range."""
        lo, hi = _PROFILE_RISK_RANGE[self.risk_profile]
        return lo <= self.overall_score <= hi


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_report(
    holdings: pd.DataFrame,
    client_id: str,
    risk_profile: Optional[str] = None,
) -> RiskReport:
    """
    Generate a full RiskReport for a client portfolio.

    Args:
        holdings:     Holdings DataFrame from mock_data.get_holdings_df().
        client_id:    Client identifier key.
        risk_profile: Override the profile from CLIENTS (optional).

    Returns:
        RiskReport dataclass with all metrics and flags.
    """
    profile = risk_profile or CLIENTS[client_id]["risk_profile"]

    # ── Component calculations ────────────────────────────────────────────────
    hhi                  = _compute_hhi(holdings)
    concentration_score  = _hhi_to_score(hhi)

    portfolio_vol        = _compute_portfolio_vol(holdings)
    volatility_score     = _vol_to_score(portfolio_vol)

    portfolio_beta       = _compute_portfolio_beta(holdings)
    beta_score           = _beta_to_score(portfolio_beta)

    alignment_score      = _compute_alignment_score(
        holdings, profile, concentration_score, volatility_score, beta_score
    )

    diversification_score = _compute_diversification_score(holdings)

    # ── Weighted composite score ───────────────────────────────────────────────
    overall = (
        0.30 * concentration_score +
        0.30 * volatility_score    +
        0.20 * beta_score          +
        0.20 * alignment_score
    )
    overall = round(min(max(overall, 0), 100), 1)

    # ── Risk band ─────────────────────────────────────────────────────────────
    band, colour = _score_to_band(overall)

    # ── Risk flags ────────────────────────────────────────────────────────────
    flags = _generate_flags(holdings, profile, hhi, portfolio_vol, portfolio_beta, overall)

    logger.debug(
        "Risk report for %s: score=%.1f (%s) | conc=%.1f vol=%.1f beta=%.1f align=%.1f",
        client_id, overall, band,
        concentration_score, volatility_score, beta_score, alignment_score,
    )

    return RiskReport(
        client_id=client_id,
        risk_profile=profile,
        overall_score=overall,
        risk_band=band,
        risk_colour=colour,
        concentration_score=round(concentration_score, 1),
        volatility_score=round(volatility_score, 1),
        beta_score=round(beta_score, 1),
        alignment_score=round(alignment_score, 1),
        hhi=round(hhi, 4),
        portfolio_beta=round(portfolio_beta, 3),
        portfolio_vol=round(portfolio_vol, 2),
        diversification_score=round(diversification_score, 1),
        flags=flags,
    )


def risk_score_only(holdings: pd.DataFrame, client_id: str) -> float:
    """Lightweight helper — returns just the composite risk score (0–100)."""
    report = compute_risk_report(holdings, client_id)
    return report.overall_score


# ─────────────────────────────────────────────────────────────────────────────
# Component calculators
# ─────────────────────────────────────────────────────────────────────────────

def _compute_hhi(holdings: pd.DataFrame) -> float:
    """
    Herfindahl-Hirschman Index = sum of squared position weights.

    HHI = 1.0  → perfectly concentrated (one holding)
    HHI ≈ 0.0  → perfectly diversified (many equal holdings)

    We compute over non-cash positions to avoid cash distorting concentration.
    """
    nc = holdings[holdings["ticker"] != "CASH"].copy()
    if nc.empty:
        return 0.0
    weights = nc["market_value"] / nc["market_value"].sum()
    return float((weights ** 2).sum())


def _hhi_to_score(hhi: float) -> float:
    """
    Map HHI (0–1) to a 0–100 risk score.
    HHI < 0.15 → well-diversified (~Low risk)
    HHI > 0.50 → highly concentrated (~Very High risk)
    """
    # Piecewise linear mapping
    if hhi <= 0.10:
        return hhi / 0.10 * 20          # 0–20
    elif hhi <= 0.25:
        return 20 + (hhi - 0.10) / 0.15 * 30   # 20–50
    elif hhi <= 0.50:
        return 50 + (hhi - 0.25) / 0.25 * 30   # 50–80
    else:
        return 80 + min((hhi - 0.50) / 0.50, 1) * 20  # 80–100


def _compute_portfolio_vol(holdings: pd.DataFrame) -> float:
    """
    Weighted-average annual volatility of the portfolio.

    Returns volatility as a decimal (e.g. 0.18 = 18%).
    This simplification assumes zero correlation between asset classes
    (conservative approximation — correlations would only reduce true vol).
    """
    total = holdings["market_value"].sum()
    if total == 0:
        return 0.0
    weights = holdings["market_value"] / total
    vols    = holdings["asset_class"].map(
        lambda c: ASSET_CLASS_METADATA.get(c, {}).get("annual_vol", 0.15)
    )
    return float((weights * vols).sum())


def _vol_to_score(vol: float) -> float:
    """
    Map annual volatility (decimal) to a 0–100 risk score.
    0% vol → 0 score | 40%+ vol → 100 score
    """
    return min(vol / 0.40 * 100, 100)


def _compute_portfolio_beta(holdings: pd.DataFrame) -> float:
    """
    Weighted portfolio beta vs. the market.
    Beta > 1.0 → more volatile than market.
    """
    total = holdings["market_value"].sum()
    if total == 0:
        return 1.0
    weights = holdings["market_value"] / total
    betas   = holdings["asset_class"].map(
        lambda c: ASSET_CLASS_METADATA.get(c, {}).get("beta", 1.0)
    )
    return float((weights * betas).sum())


def _beta_to_score(beta: float) -> float:
    """
    Map beta to 0–100 risk score.
    Beta 0.0 → 0 | Beta 2.0+ → 100
    """
    return min(beta / 2.0 * 100, 100)


def _compute_alignment_score(
    holdings: pd.DataFrame,
    risk_profile: str,
    concentration_score: float,
    volatility_score: float,
    beta_score: float,
) -> float:
    """
    Score how misaligned the portfolio is from its stated risk profile.

    Logic:
      1. Compute a proxy composite without alignment (3-component score).
      2. Find how far it sits from the profile's expected score range.
      3. Misalignment = distance from range boundary (0 if inside range).
    """
    proxy_score = 0.375 * concentration_score + 0.375 * volatility_score + 0.25 * beta_score
    lo, hi      = _PROFILE_RISK_RANGE[risk_profile]

    if proxy_score < lo:
        # Portfolio is too conservative for the stated profile
        misalignment = lo - proxy_score
    elif proxy_score > hi:
        # Portfolio is too aggressive for the stated profile
        misalignment = proxy_score - hi
    else:
        misalignment = 0

    # Scale misalignment to 0–100 (20 point distance = score of 100)
    return min(misalignment / 20 * 100, 100)


def _compute_diversification_score(holdings: pd.DataFrame) -> float:
    """
    A diversification score from 0 (fully concentrated) to 100 (well spread).

    Based on:
      - Number of asset classes (max credit: 5 classes)
      - Number of sectors covered (max credit: 7 sectors)
      - HHI inversion (1 - HHI, scaled)
    """
    nc          = holdings[holdings["ticker"] != "CASH"]
    n_classes   = nc["asset_class"].nunique()
    n_sectors   = nc["sector"].dropna().nunique()
    hhi         = _compute_hhi(holdings)

    class_score  = min(n_classes / 5, 1) * 40
    sector_score = min(n_sectors / 7, 1) * 30
    hhi_score    = (1 - hhi) * 30

    return round(class_score + sector_score + hhi_score, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Flag generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_flags(
    holdings:       pd.DataFrame,
    risk_profile:   str,
    hhi:            float,
    portfolio_vol:  float,
    portfolio_beta: float,
    overall_score:  float,
) -> list[RiskFlag]:
    """Generate a list of RiskFlag observations for the dashboard."""
    flags: list[RiskFlag] = []
    non_cash = holdings[holdings["ticker"] != "CASH"]

    # ── 1. Single-position concentration ──────────────────────────────────────
    for _, row in non_cash.iterrows():
        if row["weight"] > MAX_SINGLE_POSITION_PCT:
            flags.append(RiskFlag(
                severity="high",
                category="concentration",
                title=f"{row['ticker']} is {row['weight']:.1f}% of portfolio",
                detail=(
                    f"{row['name']} ({row['ticker']}) represents {row['weight']:.1f}% "
                    f"of total portfolio value (${row['market_value']:,.0f}). "
                    f"Consider reducing to below {MAX_SINGLE_POSITION_PCT:.0f}%."
                ),
            ))

    # ── 2. Sector concentration ────────────────────────────────────────────────
    sector_df = holdings[holdings["sector"].notna()]
    if not sector_df.empty:
        total = holdings["market_value"].sum()
        sector_weights = sector_df.groupby("sector")["market_value"].sum() / total * 100
        for sector, w in sector_weights.items():
            if w > MAX_SECTOR_PCT:
                flags.append(RiskFlag(
                    severity="medium",
                    category="concentration",
                    title=f"{sector} sector is {w:.1f}% of portfolio",
                    detail=(
                        f"The {sector} sector represents {w:.1f}% of the portfolio. "
                        f"High sector concentration increases idiosyncratic risk."
                    ),
                ))

    # ── 3. High overall volatility ────────────────────────────────────────────
    if portfolio_vol > 0.25 and risk_profile == "conservative":
        flags.append(RiskFlag(
            severity="critical",
            category="volatility",
            title=f"Portfolio volatility ({portfolio_vol*100:.1f}%) too high for profile",
            detail=(
                f"Estimated annual volatility of {portfolio_vol*100:.1f}% is unusually "
                f"high for a conservative investor. Review equity and alternative positions."
            ),
        ))
    elif portfolio_vol > 0.30 and risk_profile == "moderate":
        flags.append(RiskFlag(
            severity="high",
            category="volatility",
            title=f"Above-average portfolio volatility ({portfolio_vol*100:.1f}%)",
            detail=(
                f"Portfolio volatility of {portfolio_vol*100:.1f}% exceeds typical "
                f"moderate-profile benchmarks (~13%). Consider adding defensive positions."
            ),
        ))

    # ── 4. High beta (market sensitivity) ────────────────────────────────────
    if portfolio_beta > 1.4:
        flags.append(RiskFlag(
            severity="medium" if risk_profile == "aggressive" else "high",
            category="volatility",
            title=f"Portfolio beta is elevated ({portfolio_beta:.2f}x market)",
            detail=(
                f"A beta of {portfolio_beta:.2f} means the portfolio is expected to move "
                f"{portfolio_beta:.0%} for every 1% market move. High beta increases drawdown risk."
            ),
        ))

    # ── 5. Profile misalignment ────────────────────────────────────────────────
    lo, hi = _PROFILE_RISK_RANGE[risk_profile]
    if overall_score < lo - 5:
        flags.append(RiskFlag(
            severity="low",
            category="alignment",
            title=f"Portfolio is more conservative than the {risk_profile} profile",
            detail=(
                f"Risk score {overall_score:.0f} is below the expected range "
                f"({lo}–{hi}) for a {risk_profile} investor. The portfolio may underperform "
                f"long-term goals. Consider reviewing target allocation."
            ),
        ))
    elif overall_score > hi + 5:
        flags.append(RiskFlag(
            severity="high",
            category="alignment",
            title=f"Portfolio risk exceeds {risk_profile} profile tolerance",
            detail=(
                f"Risk score {overall_score:.0f} is above the expected range "
                f"({lo}–{hi}) for a {risk_profile} investor. The client may be taking "
                f"on more risk than intended."
            ),
        ))

    # ── 6. Low diversification ────────────────────────────────────────────────
    n_classes = non_cash["asset_class"].nunique()
    if n_classes <= 1:
        flags.append(RiskFlag(
            severity="medium",
            category="diversification",
            title="Portfolio is in a single asset class",
            detail="Holding assets across multiple classes (equity, bonds, ETFs) reduces "
                   "overall portfolio risk through diversification.",
        ))

    # ── 7. Significant unrealised loss ────────────────────────────────────────
    losers = holdings[holdings["gain_pct"] < -15]
    for _, row in losers.iterrows():
        if row["ticker"] == "CASH":
            continue
        flags.append(RiskFlag(
            severity="low",
            category="concentration",
            title=f"{row['ticker']} is down {row['gain_pct']:.1f}%",
            detail=(
                f"{row['name']} has an unrealised loss of "
                f"${abs(row['gain_loss']):,.0f} ({row['gain_pct']:.1f}%). "
                f"Review position thesis and consider tax-loss harvesting."
            ),
        ))

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_to_band(score: float) -> tuple[str, str]:
    """Return (band_label, hex_colour) for a given risk score."""
    for lo, hi, label, colour in RISK_BANDS:
        if lo <= score < hi:
            return label, colour
    return "Very High", "#ff4757"


def risk_band_for_score(score: float) -> tuple[str, str]:
    """Public wrapper — returns (band_label, colour) for a score."""
    return _score_to_band(score)
