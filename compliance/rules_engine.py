"""
compliance/rules_engine.py
───────────────────────────
Rule-based compliance engine.

Each Rule is a dataclass with a pure check() function.
Rules return RuleResult — pass/fail + explanation.
The engine runs all rules against a client's portfolio and collects violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configurable thresholds (single source of truth — edit here, affects all rules)
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS: dict = {
    # Concentration
    "max_single_position_pct":       25.0,   # Max % for any single holding
    "max_sector_pct":                50.0,   # Max % for any single sector
    "max_asset_class_pct":           80.0,   # Max % for any single asset class

    # Suitability — equity limits by risk profile
    "equity_limit_conservative":     30.0,   # Max equity % for conservative
    "equity_limit_moderate":         65.0,   # Max equity % for moderate

    # Alternative / high-risk assets
    "max_alternative_conservative":   5.0,   # Max alternatives for conservative
    "max_alternative_moderate":      15.0,   # Max alternatives for moderate

    # Cash
    "max_cash_aggressive":           15.0,   # Too much cash drag for aggressive
    "min_cash_conservative":          5.0,   # Min cash buffer for conservative

    # Diversification
    "min_asset_classes":              2,     # Minimum distinct asset classes
    "min_positions":                  3,     # Minimum non-cash positions

    # Allocation drift
    "max_drift_pct":                 15.0,   # Max deviation from target weight

    # Performance
    "max_drawdown_alert":           -20.0,   # Drawdown threshold for alert (%)

    # Senior investor (proxy — aggressive profile with very large AUM)
    "senior_aum_threshold":       500_000,   # AUM above which we check aggressiveness
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    """Result from a single rule check."""
    rule_id:     str
    rule_name:   str
    category:    str
    severity:    str        # critical | high | medium | low | info
    passed:      bool       # True = no violation
    title:       str        # Short description (used as alert title)
    explanation: str        # Human-readable detail
    data:        dict = field(default_factory=dict)   # Supporting numbers


@dataclass
class PortfolioContext:
    """
    Lightweight input bundle for the rules engine.
    Populated by AlertService before running rules.
    """
    client_id:          str
    client_name:        str
    risk_profile:       str    # conservative | moderate | aggressive
    aum:                float
    advisor_notes:      str
    holdings:           pd.DataFrame   # from mock_data.get_holdings_df()
    target_allocation:  dict           # from CLIENTS[id]["target_allocation"]
    perf_metrics:       dict           # from analytics.performance_metrics()
    rolling_returns:    dict           # from analytics.rolling_returns()
    drift_df:           pd.DataFrame   # from analytics.drift_analysis()


# ─────────────────────────────────────────────────────────────────────────────
# Rule base class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Rule:
    """
    A single compliance rule.
    Subclass and implement check(), or pass a check_fn callable.
    """
    id:          str
    name:        str
    category:    str   # concentration | suitability | exposure | diversification | performance | regulatory
    severity:    str   # severity level when violated
    description: str   # What this rule checks
    enabled:     bool = True

    def check(self, ctx: PortfolioContext) -> RuleResult:
        raise NotImplementedError

    def _pass(self, ctx: PortfolioContext, note: str = "No issues found.") -> RuleResult:
        return RuleResult(
            rule_id=self.id, rule_name=self.name, category=self.category,
            severity=self.severity, passed=True,
            title=f"{self.name} — OK", explanation=note,
        )

    def _fail(self, ctx: PortfolioContext, title: str, explanation: str, data: dict = None) -> RuleResult:
        return RuleResult(
            rule_id=self.id, rule_name=self.name, category=self.category,
            severity=self.severity, passed=False,
            title=title, explanation=explanation, data=data or {},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Individual Rules
# ─────────────────────────────────────────────────────────────────────────────

class SinglePositionConcentrationRule(Rule):
    def __init__(self):
        super().__init__(
            id="CONC-001", name="Single Position Concentration",
            category="concentration", severity="high",
            description=f"Flags any holding exceeding {THRESHOLDS['max_single_position_pct']}% of portfolio.",
        )

    def check(self, ctx: PortfolioContext) -> RuleResult:
        limit    = THRESHOLDS["max_single_position_pct"]
        nc       = ctx.holdings[ctx.holdings["ticker"] != "CASH"]
        offenders = nc[nc["weight"] > limit]

        if offenders.empty:
            return self._pass(ctx, f"No position exceeds {limit}% of portfolio.")

        lines = [f"{r['ticker']} ({r['weight']:.1f}%)" for _, r in offenders.iterrows()]
        return self._fail(
            ctx,
            title=f"{len(offenders)} position(s) exceed {limit}% weight limit",
            explanation=(
                f"The following holding(s) represent more than {limit}% of the total portfolio: "
                f"{', '.join(lines)}. High concentration in a single asset increases idiosyncratic risk. "
                f"Consider trimming or rebalancing these positions."
            ),
            data={"offenders": lines, "limit_pct": limit},
        )


class SectorConcentrationRule(Rule):
    def __init__(self):
        super().__init__(
            id="CONC-002", name="Sector Concentration",
            category="concentration", severity="medium",
            description=f"Flags any sector exceeding {THRESHOLDS['max_sector_pct']}% of portfolio.",
        )

    def check(self, ctx: PortfolioContext) -> RuleResult:
        limit    = THRESHOLDS["max_sector_pct"]
        sec_df   = ctx.holdings[ctx.holdings["sector"].notna()]
        if sec_df.empty:
            return self._pass(ctx, "No sector data available.")

        total        = ctx.holdings["market_value"].sum()
        sector_wts   = sec_df.groupby("sector")["market_value"].sum() / total * 100
        offenders    = sector_wts[sector_wts > limit]

        if offenders.empty:
            return self._pass(ctx, f"No sector exceeds {limit}% of portfolio.")

        lines = [f"{s} ({w:.1f}%)" for s, w in offenders.items()]
        return self._fail(
            ctx,
            title=f"Sector overweight: {', '.join(offenders.index.tolist())}",
            explanation=(
                f"Sector(s) {', '.join(lines)} exceed the {limit}% threshold. "
                f"Sector concentration exposes the client to industry-specific downturns. "
                f"Diversifying across additional sectors is recommended."
            ),
            data={"offenders": lines},
        )


class EquitySuitabilityRule(Rule):
    def __init__(self):
        super().__init__(
            id="SUIT-001", name="Equity Suitability Check",
            category="suitability", severity="high",
            description="Checks equity allocation is appropriate for the client's risk profile.",
        )

    def check(self, ctx: PortfolioContext) -> RuleResult:
        limits = {
            "conservative": THRESHOLDS["equity_limit_conservative"],
            "moderate":     THRESHOLDS["equity_limit_moderate"],
            "aggressive":   100.0,   # No hard cap for aggressive
        }
        limit     = limits.get(ctx.risk_profile, 100.0)
        equity_df = ctx.holdings[ctx.holdings["asset_class"] == "equity"]
        eq_pct    = equity_df["market_value"].sum() / ctx.holdings["market_value"].sum() * 100

        if eq_pct <= limit:
            return self._pass(ctx, f"Equity allocation ({eq_pct:.1f}%) within {ctx.risk_profile} limit ({limit:.0f}%).")

        return self._fail(
            ctx,
            title=f"Equity overweight for {ctx.risk_profile} profile ({eq_pct:.1f}% vs {limit:.0f}% limit)",
            explanation=(
                f"The portfolio holds {eq_pct:.1f}% in equities, exceeding the recommended "
                f"maximum of {limit:.0f}% for a {ctx.risk_profile} investor. "
                f"This may expose the client to volatility beyond their stated risk tolerance. "
                f"Consider shifting some equity weight to bonds or defensive ETFs."
            ),
            data={"equity_pct": round(eq_pct, 2), "limit_pct": limit},
        )


class AlternativeExposureRule(Rule):
    def __init__(self):
        super().__init__(
            id="SUIT-002", name="High-Risk Alternative Exposure",
            category="exposure", severity="critical",
            description="Flags excessive alternative asset exposure for conservative/moderate profiles.",
        )

    def check(self, ctx: PortfolioContext) -> RuleResult:
        limits = {
            "conservative": THRESHOLDS["max_alternative_conservative"],
            "moderate":     THRESHOLDS["max_alternative_moderate"],
            "aggressive":   100.0,
        }
        limit  = limits.get(ctx.risk_profile, 100.0)
        alt_df = ctx.holdings[ctx.holdings["asset_class"] == "alternative"]
        alt_pct = alt_df["market_value"].sum() / ctx.holdings["market_value"].sum() * 100

        if alt_pct <= limit:
            return self._pass(ctx, f"Alternative exposure ({alt_pct:.1f}%) within limit ({limit:.0f}%).")

        tickers = alt_df["ticker"].tolist()
        return self._fail(
            ctx,
            title=f"Alternative/crypto exposure ({alt_pct:.1f}%) exceeds {ctx.risk_profile} limit ({limit:.0f}%)",
            explanation=(
                f"Positions in high-risk alternative assets ({', '.join(tickers)}) total {alt_pct:.1f}% "
                f"of the portfolio. For a {ctx.risk_profile} investor, the recommended maximum is {limit:.0f}%. "
                f"Alternative assets can be highly volatile and illiquid. Immediate review recommended."
            ),
            data={"alternative_pct": round(alt_pct, 2), "limit_pct": limit, "tickers": tickers},
        )


class CashDragRule(Rule):
    def __init__(self):
        super().__init__(
            id="SUIT-003", name="Cash Drag Check",
            category="suitability", severity="low",
            description="Flags excessive cash for growth profiles, or insufficient cash buffer for conservative.",
        )

    def check(self, ctx: PortfolioContext) -> RuleResult:
        cash_df  = ctx.holdings[ctx.holdings["asset_class"] == "cash"]
        cash_pct = cash_df["market_value"].sum() / ctx.holdings["market_value"].sum() * 100

        if ctx.risk_profile == "aggressive" and cash_pct > THRESHOLDS["max_cash_aggressive"]:
            return self._fail(
                ctx,
                title=f"Cash drag: {cash_pct:.1f}% cash in aggressive portfolio",
                explanation=(
                    f"An aggressive portfolio holding {cash_pct:.1f}% in cash may significantly "
                    f"drag long-term returns. Cash should typically be below "
                    f"{THRESHOLDS['max_cash_aggressive']:.0f}% for growth-oriented investors. "
                    f"Consider deploying excess cash into growth assets."
                ),
                data={"cash_pct": round(cash_pct, 2)},
            )

        if ctx.risk_profile == "conservative" and cash_pct < THRESHOLDS["min_cash_conservative"]:
            return self._fail(
                ctx,
                title=f"Insufficient cash buffer ({cash_pct:.1f}%) for conservative profile",
                explanation=(
                    f"Conservative investors should maintain at least "
                    f"{THRESHOLDS['min_cash_conservative']:.0f}% in cash or equivalents "
                    f"as a liquidity buffer. Current cash is only {cash_pct:.1f}%."
                ),
                data={"cash_pct": round(cash_pct, 2)},
            )

        return self._pass(ctx, f"Cash allocation ({cash_pct:.1f}%) is appropriate.")


class DiversificationRule(Rule):
    def __init__(self):
        super().__init__(
            id="DIV-001", name="Minimum Diversification",
            category="diversification", severity="medium",
            description="Ensures the portfolio holds sufficient asset classes and positions.",
        )

    def check(self, ctx: PortfolioContext) -> RuleResult:
        nc         = ctx.holdings[ctx.holdings["ticker"] != "CASH"]
        n_classes  = nc["asset_class"].nunique()
        n_positions = len(nc)
        min_classes = THRESHOLDS["min_asset_classes"]
        min_pos     = THRESHOLDS["min_positions"]

        issues = []
        if n_classes < min_classes:
            issues.append(f"only {n_classes} asset class(es) (minimum {min_classes})")
        if n_positions < min_pos:
            issues.append(f"only {n_positions} position(s) (minimum {min_pos})")

        if not issues:
            return self._pass(ctx, f"Portfolio has {n_positions} positions across {n_classes} asset classes.")

        return self._fail(
            ctx,
            title=f"Under-diversified portfolio: {'; '.join(issues)}",
            explanation=(
                f"The portfolio is insufficiently diversified — {', '.join(issues)}. "
                f"Diversification across multiple asset classes reduces unsystematic risk "
                f"and smooths long-term returns."
            ),
            data={"n_classes": n_classes, "n_positions": n_positions},
        )


class AllocationDriftRule(Rule):
    def __init__(self):
        super().__init__(
            id="RBAL-001", name="Allocation Drift",
            category="rebalancing", severity="medium",
            description=f"Flags drift from target allocation exceeding {THRESHOLDS['max_drift_pct']}%.",
        )

    def check(self, ctx: PortfolioContext) -> RuleResult:
        limit    = THRESHOLDS["max_drift_pct"]
        drift_df = ctx.drift_df
        if drift_df.empty:
            return self._pass(ctx, "No target allocation defined.")

        over = drift_df[drift_df["drift"].abs() > limit]
        if over.empty:
            return self._pass(ctx, f"All asset classes within {limit}% of target.")

        lines = [f"{r['asset_class']} ({r['drift']:+.1f}%)" for _, r in over.iterrows()]
        return self._fail(
            ctx,
            title=f"Rebalancing needed: {len(over)} class(es) drifted >{limit}%",
            explanation=(
                f"The following asset classes have drifted beyond {limit}% from their target: "
                f"{', '.join(lines)}. Portfolio drift can alter the client's effective risk profile "
                f"and should be corrected through rebalancing."
            ),
            data={"drifted_classes": lines, "limit_pct": limit},
        )


class SeniorInvestorWarningRule(Rule):
    def __init__(self):
        super().__init__(
            id="REG-001", name="Senior Investor Suitability Warning",
            category="regulatory", severity="critical",
            description=(
                "Flags aggressive risk profiles for high-AUM clients who may be "
                "approaching or at retirement age based on portfolio characteristics."
            ),
        )

    def check(self, ctx: PortfolioContext) -> RuleResult:
        # Proxy heuristic: very large AUM + aggressive profile + high alternative exposure
        # (In a real system this would use actual DOB from a CRM)
        alt_df  = ctx.holdings[ctx.holdings["asset_class"] == "alternative"]
        alt_pct = alt_df["market_value"].sum() / ctx.holdings["market_value"].sum() * 100
        is_high_aum     = ctx.aum >= THRESHOLDS["senior_aum_threshold"]
        is_aggressive   = ctx.risk_profile == "aggressive"
        has_high_alt    = alt_pct > 10

        if not (is_high_aum and is_aggressive and has_high_alt):
            return self._pass(ctx, "No senior investor suitability concerns flagged.")

        return self._fail(
            ctx,
            title="Senior investor suitability review required",
            explanation=(
                f"Client {ctx.client_name} has a large portfolio (${ctx.aum:,.0f}) with an "
                f"aggressive risk profile and {alt_pct:.1f}% in high-risk alternative assets. "
                f"Regulatory best-practice (e.g. FINRA Rule 4512) requires advisors to re-confirm "
                f"suitability for clients who may be near or at retirement age. "
                f"Please verify the client's current risk tolerance, time horizon, and liquidity needs."
            ),
            data={"aum": ctx.aum, "alt_pct": round(alt_pct, 2)},
        )


class DrawdownAlertRule(Rule):
    def __init__(self):
        super().__init__(
            id="PERF-001", name="Significant Drawdown Alert",
            category="performance", severity="high",
            description=f"Alerts when 52-week max drawdown exceeds {abs(THRESHOLDS['max_drawdown_alert'])}%.",
        )

    def check(self, ctx: PortfolioContext) -> RuleResult:
        threshold = THRESHOLDS["max_drawdown_alert"]
        dd        = ctx.perf_metrics.get("max_drawdown_pct", 0)

        if dd >= threshold:   # threshold is negative, so >= means less severe
            return self._pass(ctx, f"Max drawdown ({dd:.1f}%) within acceptable range.")

        return self._fail(
            ctx,
            title=f"Significant drawdown: {dd:.1f}% peak-to-trough",
            explanation=(
                f"The portfolio experienced a maximum drawdown of {dd:.1f}% over the past 52 weeks. "
                f"This exceeds the alert threshold of {abs(threshold):.0f}%. "
                f"For a {ctx.risk_profile} investor, this level of drawdown warrants a portfolio review "
                f"and a client conversation about risk tolerance and time horizon."
            ),
            data={"max_drawdown_pct": dd, "threshold_pct": threshold},
        )


class UnderperformanceRule(Rule):
    def __init__(self):
        super().__init__(
            id="PERF-002", name="Rolling Return Underperformance",
            category="performance", severity="low",
            description="Flags portfolios with negative returns across multiple time periods.",
        )

    def check(self, ctx: PortfolioContext) -> RuleResult:
        rolling  = ctx.rolling_returns
        negative = {k: v for k, v in rolling.items() if v is not None and v < 0}

        if len(negative) < 2:
            return self._pass(ctx, "Portfolio returns are acceptable across rolling periods.")

        lines = [f"{period}: {ret:+.1f}%" for period, ret in negative.items()]
        return self._fail(
            ctx,
            title=f"Negative returns across {len(negative)} rolling periods",
            explanation=(
                f"The portfolio shows negative performance across multiple timeframes: "
                f"{', '.join(lines)}. Persistent underperformance may indicate structural issues "
                f"with the portfolio composition or a need to review strategy alignment."
            ),
            data={"negative_periods": negative},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rules Engine
# ─────────────────────────────────────────────────────────────────────────────

class RulesEngine:
    """
    Runs all enabled compliance rules against a PortfolioContext.

    Usage:
        engine  = RulesEngine()
        results = engine.run(ctx)
        violations = [r for r in results if not r.passed]
    """

    def __init__(self, rules: Optional[list[Rule]] = None) -> None:
        self.rules: list[Rule] = rules or _default_rules()

    def run(self, ctx: PortfolioContext) -> list[RuleResult]:
        """Run all enabled rules and return results (pass and fail)."""
        results: list[RuleResult] = []
        for rule in self.rules:
            if not rule.enabled:
                continue
            try:
                result = rule.check(ctx)
                results.append(result)
                if not result.passed:
                    logger.info(
                        "VIOLATION [%s] %s | client=%s | %s",
                        result.severity.upper(), result.rule_id,
                        ctx.client_name, result.title,
                    )
            except Exception as exc:
                logger.error("Rule %s failed with exception: %s", rule.id, exc)

        return results

    def violations(self, ctx: PortfolioContext) -> list[RuleResult]:
        """Convenience — return only failing rules."""
        return [r for r in self.run(ctx) if not r.passed]

    @property
    def rule_count(self) -> int:
        return len([r for r in self.rules if r.enabled])


def _default_rules() -> list[Rule]:
    """Return the full default rule set."""
    return [
        SinglePositionConcentrationRule(),
        SectorConcentrationRule(),
        EquitySuitabilityRule(),
        AlternativeExposureRule(),
        CashDragRule(),
        DiversificationRule(),
        AllocationDriftRule(),
        SeniorInvestorWarningRule(),
        DrawdownAlertRule(),
        UnderperformanceRule(),
    ]


def get_all_rules() -> list[Rule]:
    """Public accessor for the rule registry (used by the dashboard)."""
    return _default_rules()
