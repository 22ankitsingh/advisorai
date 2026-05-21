"""
services/summary_service.py
────────────────────────────
AI-powered client summary generation service.

Responsibilities:
  - Assemble a structured ClientContext from all available data sources
  - Build modular Gemini prompt templates for four summary types
  - Call Gemini (or return a static fallback if key is missing)
  - Return typed SummaryResult objects ready for the UI to render

Four summary types:
  1. meeting_prep     → Full advisor briefing before a client meeting
  2. risk_explanation → Plain-English explanation of the client's risk position
  3. portfolio_insights → Key observations and anomalies in the portfolio
  4. behavioral_profile → Investment behaviour and personality profile

Design principle:
  All Gemini prompting is isolated here. The UI page just calls generate()
  and receives a SummaryResult — it never builds prompts or calls Gemini directly.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from portfolio.mock_data import (
    CLIENTS, ASSET_CLASS_METADATA, RISK_FREE_RATE,
    get_holdings_df, get_nav_history,
)
from portfolio.analytics import (
    portfolio_summary, allocation_by_class, allocation_by_sector,
    performance_metrics, rolling_returns, drift_analysis, top_holdings,
)
from portfolio.risk_engine import compute_risk_report, RiskReport
from services.gemini_service import GeminiService, GeminiServiceError, is_ai_available
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Summary type registry
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_TYPES: dict[str, dict] = {
    "meeting_prep": {
        "label":       "Meeting Preparation Brief",
        "icon":        "📋",
        "description": "Full advisor briefing — portfolio state, risks, talking points, and action items.",
    },
    "risk_explanation": {
        "label":       "Risk Profile Explanation",
        "icon":        "⚖️",
        "description": "Plain-English explanation of portfolio risk, suitable to share with the client.",
    },
    "portfolio_insights": {
        "label":       "Portfolio Insights",
        "icon":        "🔍",
        "description": "Key observations, anomalies, and opportunities detected in the portfolio.",
    },
    "behavioral_profile": {
        "label":       "Investment Behaviour Profile",
        "icon":        "🧠",
        "description": "Client investment personality, typical behavioural biases, and engagement style.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClientContext:
    """
    Fully assembled snapshot of a client's financial situation.
    Built once and passed to any prompt builder.
    """
    client_id:        str
    name:             str
    email:            str
    risk_profile:     str
    advisor_notes:    str

    # Portfolio
    holdings:         pd.DataFrame
    summary:          dict              # from portfolio_summary()
    alloc_class:      pd.DataFrame      # from allocation_by_class()
    alloc_sector:     pd.DataFrame      # from allocation_by_sector()
    top5:             pd.DataFrame      # top 5 holdings
    drift:            pd.DataFrame      # actual vs. target

    # Performance
    perf_metrics:     dict              # from performance_metrics()
    rolling:          dict              # from rolling_returns()

    # Risk
    risk_report:      RiskReport

    # Meta
    generated_on:     str = field(default_factory=lambda: date.today().isoformat())


@dataclass
class SummaryResult:
    """Output of a summary generation call."""
    summary_type:   str
    client_name:    str
    content:        str          # The generated summary text (Markdown)
    is_ai_generated: bool        # False if Gemini unavailable (template fallback)
    error:          str  = ""    # Non-empty if generation failed
    generated_on:   str  = field(default_factory=lambda: date.today().isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def build_client_context(client_id: str) -> ClientContext:
    """
    Assemble a complete ClientContext for a given client_id.

    Phase 6: Holdings are sourced from the DB first.
    Falls back to mock_data.get_holdings_df() for legacy mock_key clients
    when DB holdings are absent. NAV history always comes from mock_data
    (no NAV table in DB yet).

    Args:
        client_id: mock_data key OR db:<int> for DB-only clients.

    Returns:
        ClientContext with all analytics pre-computed.
    """
    # ── Resolve client metadata ───────────────────────────────────────────────────────────
    if client_id.startswith("db:"):
        # DB-only client
        db_id = int(client_id[3:])
        mock_key = None
        from database.repositories.client_repository import ClientRepository
        db_client = ClientRepository().get_by_id(db_id) or {}
        name          = db_client.get("name", "Unknown")
        email         = db_client.get("email", "")
        risk_profile  = db_client.get("risk_profile", "moderate")
        advisor_notes = db_client.get("advisor_notes", "")
        target: dict  = {}
    else:
        # Legacy mock_data client
        mock_key = client_id
        db_id    = None
        client   = CLIENTS[client_id]
        name          = client["name"]
        email         = client.get("email", "")
        risk_profile  = client["risk_profile"]
        advisor_notes = client.get("advisor_notes", "")
        target        = client["target_allocation"]

    # ── Holdings: DB first, mock_data fallback ───────────────────────────────────────
    from database.repositories.portfolio_repository import PortfolioRepository
    pr        = PortfolioRepository()
    holdings  = pd.DataFrame()

    if db_id:
        ports = pr.get_for_client(db_id)
        if ports:
            holdings = pr.get_holdings_df(ports[0]["id"])
    elif mock_key:
        ports = pr.get_for_client_by_mock_key(mock_key)
        if ports:
            holdings = pr.get_holdings_df(ports[0]["id"])

    if holdings.empty and mock_key:
        holdings = get_holdings_df(mock_key)

    # ── NAV history (mock_data or empty) ───────────────────────────────────────
    nav_df = pd.DataFrame()
    if mock_key:
        nav_df = get_nav_history(mock_key, weeks=52)

    # ── Target allocation fallback ───────────────────────────────────────────────
    if not target and not holdings.empty:
        classes = holdings["asset_class"].unique()
        share   = round(100 / len(classes), 1)
        target  = {c: share for c in classes}

    # ── Risk report ───────────────────────────────────────────────────────────────
    rr = compute_risk_report(
        holdings if not holdings.empty else pd.DataFrame(
            columns=["ticker","asset_class","market_value","cost_basis",
                     "gain_loss","gain_pct","weight","sector"]
        ),
        mock_key or "db_client",
        risk_profile=risk_profile if not mock_key else None,
    )

    return ClientContext(
        client_id=client_id,
        name=name,
        email=email,
        risk_profile=risk_profile,
        advisor_notes=advisor_notes,
        holdings=holdings,
        summary=portfolio_summary(holdings) if not holdings.empty else
            {"total_value":0,"total_cost":0,"total_gain_loss":0,"total_gain_pct":0,
             "num_positions":0,"num_asset_classes":0,"largest_position_weight":0,"cash_weight":0},
        alloc_class=allocation_by_class(holdings) if not holdings.empty else pd.DataFrame(),
        alloc_sector=allocation_by_sector(holdings) if not holdings.empty else pd.DataFrame(),
        top5=top_holdings(holdings, n=5) if not holdings.empty else pd.DataFrame(),
        drift=drift_analysis(holdings, target) if not holdings.empty and target else pd.DataFrame(),
        perf_metrics=performance_metrics(nav_df) if not nav_df.empty else {},
        rolling=rolling_returns(nav_df) if not nav_df.empty else {},
        risk_report=rr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders — one per summary type
# ─────────────────────────────────────────────────────────────────────────────

def _format_holdings_table(ctx: ClientContext) -> str:
    """Render top holdings as a plain-text table for prompt injection."""
    lines = ["Ticker | Name                    | Class       | Weight% | Gain%"]
    lines.append("-------|-------------------------|-------------|---------|-------")
    for _, row in ctx.top5.iterrows():
        lines.append(
            f"{row['ticker']:<7}| {row['name'][:25]:<25}| {row['asset_class']:<12}| "
            f"{row['weight']:>6.1f}% | {row['gain_pct']:>+6.1f}%"
        )
    return "\n".join(lines)


def _format_drift_table(ctx: ClientContext) -> str:
    """Render allocation drift as a plain-text table."""
    lines = ["Asset Class  | Actual% | Target% | Drift%  | Status"]
    lines.append("-------------|---------|---------|---------|------------")
    for _, row in ctx.drift.iterrows():
        lines.append(
            f"{row['asset_class']:<13}| {row['actual_weight']:>6.1f}% | "
            f"{row['target_weight']:>6.1f}% | {row['drift']:>+6.1f}% | {row['status']}"
        )
    return "\n".join(lines)


def _shared_data_block(ctx: ClientContext) -> str:
    """Shared data facts injected into every prompt."""
    s   = ctx.summary
    rr  = ctx.risk_report
    pm  = ctx.perf_metrics
    rol = ctx.rolling

    return textwrap.dedent(f"""
        CLIENT: {ctx.name}
        RISK PROFILE: {ctx.risk_profile.title()}
        ADVISOR NOTES: {ctx.advisor_notes}

        PORTFOLIO OVERVIEW:
          Total Value:      ${s['total_value']:>14,.2f}
          Cost Basis:       ${s['total_cost']:>14,.2f}
          Total Gain/Loss:  ${s['total_gain_loss']:>+14,.2f} ({s['total_gain_pct']:+.1f}%)
          Positions:        {s['num_positions']}
          Asset Classes:    {s['num_asset_classes']}
          Cash Weight:      {s['cash_weight']:.1f}%
          Largest Position: {s['largest_position_weight']:.1f}%

        PERFORMANCE (52-week):
          Total Return:     {pm.get('total_return_pct', 0):+.1f}%
          Annualised:       {pm.get('annualised_return', 0):+.1f}%
          Volatility:       {pm.get('volatility_annual', 0):.1f}%
          Sharpe Ratio:     {pm.get('sharpe_ratio', 0):.2f}
          Max Drawdown:     {pm.get('max_drawdown_pct', 0):.1f}%
          Best Week:        {pm.get('best_week_pct', 0):+.1f}%
          Worst Week:       {pm.get('worst_week_pct', 0):+.1f}%

        ROLLING RETURNS:
          4-Week:   {rol.get('4W', 'N/A')}%   13-Week: {rol.get('13W', 'N/A')}%
          26-Week:  {rol.get('26W', 'N/A')}%  52-Week: {rol.get('52W', 'N/A')}%

        RISK ASSESSMENT:
          Overall Score:    {rr.overall_score:.1f}/100 ({rr.risk_band})
          Concentration:    {rr.concentration_score:.1f}/100 (HHI={rr.hhi:.3f})
          Volatility Score: {rr.volatility_score:.1f}/100 (Portfolio Vol={rr.portfolio_vol:.1f}%)
          Beta Score:       {rr.beta_score:.1f}/100 (Portfolio Beta={rr.portfolio_beta:.2f})
          Alignment Score:  {rr.alignment_score:.1f}/100 (Aligned={rr.is_aligned})
          Diversification:  {rr.diversification_score:.1f}/100

        TOP 5 HOLDINGS:
        {_format_holdings_table(ctx)}

        ALLOCATION vs. TARGET:
        {_format_drift_table(ctx)}

        RISK FLAGS ({len(rr.flags)}):
        {chr(10).join(f'  [{f.severity.upper()}] {f.title}: {f.detail}' for f in rr.flags) or '  None'}
    """).strip()


def _build_meeting_prep_prompt(ctx: ClientContext) -> str:
    return textwrap.dedent(f"""
        You are a senior financial advisor assistant. Using the data below, write a professional
        MEETING PREPARATION BRIEF for an advisor who is about to meet with their client.

        Format the output in Markdown with clear sections. Be specific, data-driven, and actionable.
        Do not invent data — only use what is provided. Use professional advisory language.

        Required sections:
        1. **Executive Summary** (3–4 sentences — the most important things to know)
        2. **Portfolio Health** (current state, strengths and concerns)
        3. **Performance Commentary** (explain the returns in context)
        4. **Risk Assessment** (interpret the risk score and flags in plain English)
        5. **Allocation Drift** (where is the portfolio off-target and why it matters)
        6. **Key Discussion Points** (3–5 bullet points the advisor should raise)
        7. **Recommended Action Items** (specific, prioritised next steps)
        8. **Talking Points for the Client** (plain-English phrases the advisor can use)

        --- CLIENT DATA ---
        {_shared_data_block(ctx)}
        --- END DATA ---

        Write the Meeting Preparation Brief now:
    """).strip()


def _build_risk_explanation_prompt(ctx: ClientContext) -> str:
    return textwrap.dedent(f"""
        You are a financial advisor helping explain portfolio risk to a client.

        Write a RISK PROFILE EXPLANATION that:
        - Is written directly TO the client (use "your portfolio", "you")
        - Uses plain English — avoid jargon, or explain it when necessary
        - Explains what the risk score means in practical terms
        - Describes what could go wrong and what protects them
        - Mentions the specific risk flags found in their portfolio
        - Ends with reassurance and context appropriate to their risk profile
        - Is between 300–500 words
        - Uses Markdown formatting (headers, bullets)

        --- CLIENT DATA ---
        {_shared_data_block(ctx)}
        --- END DATA ---

        Write the Risk Profile Explanation now:
    """).strip()


def _build_portfolio_insights_prompt(ctx: ClientContext) -> str:
    return textwrap.dedent(f"""
        You are a portfolio analyst writing an INSIGHTS REPORT for a financial advisor.

        Analyse the portfolio data below and produce a structured insights report with:
        1. **Standout Observations** — what is unusual or noteworthy
        2. **Winners & Opportunities** — best performing positions and why they matter
        3. **Underperformers & Risks** — positions dragging performance
        4. **Concentration Analysis** — any overweights that need attention
        5. **Diversification Gaps** — what's missing or underrepresented
        6. **Rebalancing Opportunities** — specific trades or shifts to consider
        7. **Short-term vs Long-term Outlook** — based on portfolio composition

        Be specific — reference actual tickers, weights, and return figures from the data.
        Do not fabricate information. Use Markdown. Be concise and analytical.

        --- CLIENT DATA ---
        {_shared_data_block(ctx)}
        --- END DATA ---

        Write the Portfolio Insights Report now:
    """).strip()


def _build_behavioral_profile_prompt(ctx: ClientContext) -> str:
    return textwrap.dedent(f"""
        You are a behavioural finance expert writing an INVESTMENT BEHAVIOUR PROFILE
        for a financial advisor to better understand and serve their client.

        Based on the client's risk profile, portfolio composition, and advisor notes,
        write a thoughtful profile covering:
        1. **Investor Personality Type** — e.g. Growth Seeker, Income Defender, Balanced Builder
        2. **Likely Decision-Making Style** — how they probably react to market events
        3. **Common Behavioural Biases to Watch For** — (e.g. overconfidence, loss aversion)
        4. **Communication Preferences** — how to frame discussions for this type of client
        5. **Emotional Triggers** — what market events might cause them to act irrationally
        6. **Advisor Tips** — specific advice for working with this client personality

        Base inferences on the data provided. Clearly distinguish observation from inference.
        Be empathetic, professional, and constructive. Use Markdown. 300–450 words.

        --- CLIENT DATA ---
        {_shared_data_block(ctx)}
        --- END DATA ---

        Write the Investment Behaviour Profile now:
    """).strip()


_PROMPT_BUILDERS = {
    "meeting_prep":      _build_meeting_prep_prompt,
    "risk_explanation":  _build_risk_explanation_prompt,
    "portfolio_insights": _build_portfolio_insights_prompt,
    "behavioral_profile": _build_behavioral_profile_prompt,
}


# ─────────────────────────────────────────────────────────────────────────────
# Static fallbacks — used when Gemini is unavailable
# ─────────────────────────────────────────────────────────────────────────────

def _static_meeting_prep(ctx: ClientContext) -> str:
    s  = ctx.summary
    rr = ctx.risk_report
    pm = ctx.perf_metrics
    flags_text = "\n".join(
        f"- **[{f.severity.upper()}]** {f.title}" for f in rr.flags
    ) or "- No active risk flags."

    top_ticker = ctx.top5.iloc[0] if not ctx.top5.empty else None
    top_line   = (
        f"{top_ticker['ticker']} ({top_ticker['weight']:.1f}%)"
        if top_ticker is not None else "N/A"
    )

    return textwrap.dedent(f"""
        # Meeting Preparation Brief — {ctx.name}
        *Generated: {ctx.generated_on} | Template mode (AI unavailable)*

        ---

        ## Executive Summary
        {ctx.name} holds a **{ctx.risk_profile}** risk profile with a portfolio currently valued at
        **${s['total_value']:,.0f}**, reflecting an overall gain/loss of **${s['total_gain_loss']:+,.0f}**
        ({s['total_gain_pct']:+.1f}%) on invested capital. The portfolio spans **{s['num_positions']}
        positions** across **{s['num_asset_classes']} asset classes**. The 52-week risk score is
        **{rr.overall_score:.0f}/100 ({rr.risk_band})**.

        ---

        ## Portfolio Health
        | Metric               | Value                          |
        |----------------------|-------------------------------|
        | Portfolio Value      | ${s['total_value']:,.0f}       |
        | Cost Basis           | ${s['total_cost']:,.0f}        |
        | Total Gain / Loss    | ${s['total_gain_loss']:+,.0f}  |
        | Return               | {s['total_gain_pct']:+.1f}%    |
        | Largest Position     | {top_line}                     |
        | Cash Weight          | {s['cash_weight']:.1f}%        |

        ---

        ## Risk Assessment
        - **Overall Risk Score:** {rr.overall_score:.1f}/100 — **{rr.risk_band}**
        - **Portfolio Volatility:** {rr.portfolio_vol:.1f}% annually
        - **Portfolio Beta:** {rr.portfolio_beta:.2f}x market
        - **Diversification Score:** {rr.diversification_score:.1f}/100
        - **Profile Aligned:** {'Yes' if rr.is_aligned else 'No — review allocation'}

        ### Active Risk Flags
        {flags_text}

        ---

        ## Performance (52-Week)
        | Period   | Return         |
        |----------|----------------|
        | 4-Week   | {ctx.rolling.get('4W', 'N/A')}%   |
        | 13-Week  | {ctx.rolling.get('13W', 'N/A')}%  |
        | 26-Week  | {ctx.rolling.get('26W', 'N/A')}%  |
        | 52-Week  | {ctx.rolling.get('52W', 'N/A')}%  |
        | Sharpe Ratio | {pm.get('sharpe_ratio', 'N/A')} |
        | Max Drawdown | {pm.get('max_drawdown_pct', 'N/A')}% |

        ---

        ## Advisor Notes
        > {ctx.advisor_notes or 'No notes on file.'}

        ---

        ## Suggested Action Items
        1. Review risk flag(s) and discuss with client
        2. {'Rebalance overweight positions' if not rr.is_aligned else 'Confirm allocation remains on-target'}
        3. Discuss upcoming market outlook relative to client's risk profile
        4. Update advisor notes after meeting

        ---
        *⚠️ Add your GEMINI_API_KEY to .env to enable AI-generated summaries.*
    """).strip()


def _static_fallback(ctx: ClientContext, summary_type: str) -> str:
    """Simple fallback for non-meeting-prep types when AI is unavailable."""
    if summary_type == "meeting_prep":
        return _static_meeting_prep(ctx)

    type_info = SUMMARY_TYPES.get(summary_type, {})
    return textwrap.dedent(f"""
        # {type_info.get('icon', '')} {type_info.get('label', summary_type)} — {ctx.name}
        *Generated: {ctx.generated_on} | Template mode (AI unavailable)*

        ---

        This summary type requires a Gemini API key to generate AI content.

        **To enable AI summaries:**
        1. Get a free key at https://aistudio.google.com/app/apikey
        2. Add it to your `.env` file: `GEMINI_API_KEY=your_key_here`
        3. Restart the app

        **Available data for {ctx.name}:**
        - Portfolio Value: ${ctx.summary['total_value']:,.0f}
        - Risk Score: {ctx.risk_report.overall_score:.1f}/100 ({ctx.risk_report.risk_band})
        - Risk Profile: {ctx.risk_profile.title()}
        - Active Flags: {len(ctx.risk_report.flags)}

        *The meeting preparation brief is available without an API key.*
    """).strip()


# ─────────────────────────────────────────────────────────────────────────────
# SummaryService
# ─────────────────────────────────────────────────────────────────────────────

class SummaryService:
    """
    Generates AI-powered client summaries using Gemini.

    Usage:
        service = SummaryService()
        result  = service.generate("sarah_mitchell", "meeting_prep")
        print(result.content)
    """

    def __init__(self) -> None:
        self._ai_available = is_ai_available()
        self._gemini: Optional[GeminiService] = None   # Lazy-init per generation

        logger.info(
            "SummaryService initialised | AI available: %s", self._ai_available
        )

    def generate(
        self,
        client_id:    str,
        summary_type: str,
    ) -> SummaryResult:
        """
        Generate a summary for a client.

        Args:
            client_id:    Client identifier (key in CLIENTS).
            summary_type: One of: meeting_prep, risk_explanation,
                          portfolio_insights, behavioral_profile.

        Returns:
            SummaryResult with content and metadata.
        """
        if summary_type not in _PROMPT_BUILDERS:
            raise ValueError(
                f"Unknown summary_type: {summary_type!r}. "
                f"Valid: {list(_PROMPT_BUILDERS)}"
            )

        logger.info(
            "Generating %s for %s (AI=%s)", summary_type, client_id, self._ai_available
        )

        # Assemble context (analytics run here, once per call)
        ctx = build_client_context(client_id)

        # Use AI if available, otherwise fall back to template
        if not self._ai_available:
            logger.info("Gemini unavailable — using static template.")
            content = _static_fallback(ctx, summary_type)
            return SummaryResult(
                summary_type=summary_type,
                client_name=ctx.name,
                content=content,
                is_ai_generated=False,
            )

        # Build the prompt
        prompt_fn = _PROMPT_BUILDERS[summary_type]
        prompt    = prompt_fn(ctx)

        # Each summary gets its own fresh Gemini session with a focused system instruction
        try:
            gemini = GeminiService(
                client_context={
                    "name":          ctx.name,
                    "risk_profile":  ctx.risk_profile,
                    "aum":           ctx.summary["total_value"],
                    "advisor_notes": ctx.advisor_notes,
                }
            )
            content = gemini.chat(prompt)

            return SummaryResult(
                summary_type=summary_type,
                client_name=ctx.name,
                content=content,
                is_ai_generated=True,
            )

        except GeminiServiceError as exc:
            logger.error("Gemini failed for %s/%s: %s", client_id, summary_type, exc)
            # Degrade gracefully — return template with error note
            fallback = _static_fallback(ctx, summary_type)
            error_note = f"\n\n---\n⚠️ *AI generation failed: {exc}*"
            return SummaryResult(
                summary_type=summary_type,
                client_name=ctx.name,
                content=fallback + error_note,
                is_ai_generated=False,
                error=str(exc),
            )

        except Exception as exc:
            logger.error("Unexpected error generating summary: %s", exc)
            return SummaryResult(
                summary_type=summary_type,
                client_name=ctx.name,
                content=f"# Error\n\nAn unexpected error occurred: {exc}",
                is_ai_generated=False,
                error=str(exc),
            )

    def get_context(self, client_id: str) -> ClientContext:
        """
        Public access to the assembled ClientContext.
        Useful for the UI to display the data used as input.
        """
        return build_client_context(client_id)
