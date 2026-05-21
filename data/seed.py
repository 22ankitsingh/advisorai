"""
data/seed.py
─────────────
Mock data seeder.

Populates the database with realistic-looking fictional financial data
so the UI has something to display from day one.

Run directly:  python -m data.seed
Or imported:   from data.seed import seed_all
"""

import sys
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.database import get_db_connection, init_database
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Mock clients ──────────────────────────────────────────────────────────────

CLIENTS = [
    {
        "name": "Sarah Mitchell",
        "email": "sarah.mitchell@example.com",
        "phone": "+1-555-0101",
        "risk_profile": "aggressive",
        "aum": 850_000.00,
        "advisor_notes": "Tech entrepreneur. Interested in growth stocks and early-stage investments.",
    },
    {
        "name": "Robert Chen",
        "email": "robert.chen@example.com",
        "phone": "+1-555-0102",
        "risk_profile": "moderate",
        "aum": 420_000.00,
        "advisor_notes": "Software engineer, mid-career. Balanced growth and income focus.",
    },
    {
        "name": "Eleanor Vasquez",
        "email": "eleanor.vasquez@example.com",
        "phone": "+1-555-0103",
        "risk_profile": "conservative",
        "aum": 1_200_000.00,
        "advisor_notes": "Retired school principal. Capital preservation is top priority.",
    },
    {
        "name": "Marcus Thompson",
        "email": "marcus.thompson@example.com",
        "phone": "+1-555-0104",
        "risk_profile": "moderate",
        "aum": 280_000.00,
        "advisor_notes": "Small business owner. Wants to diversify outside of his business.",
    },
    {
        "name": "Priya Kapoor",
        "email": "priya.kapoor@example.com",
        "phone": "+1-555-0105",
        "risk_profile": "aggressive",
        "aum": 560_000.00,
        "advisor_notes": "Physician. High income, long time horizon, interested in healthcare sector.",
    },
]

# ── Mock holdings per client (index matches CLIENTS list) ─────────────────────

HOLDINGS_PER_CLIENT = [
    # Sarah Mitchell — aggressive, tech-heavy
    [
        ("NVDA", "NVIDIA Corporation",      "equity", 120,  410.0, 875.50,  "Technology"),
        ("TSLA", "Tesla Inc.",              "equity", 80,   180.0, 245.30,  "Consumer Discretionary"),
        ("META", "Meta Platforms",          "equity", 150,  290.0, 480.20,  "Technology"),
        ("ARKK", "ARK Innovation ETF",      "etf",    500,  45.0,  52.10,   None),
        ("BTC",  "Bitcoin (via ETF)",       "alternative", 3, 28000.0, 62000.0, None),
        ("CASH", "Cash & Equivalents",      "cash",   1,    50000.0, 50000.0, None),
    ],
    # Robert Chen — moderate, balanced
    [
        ("AAPL", "Apple Inc.",              "equity", 100,  155.0, 189.40,  "Technology"),
        ("VTI",  "Vanguard Total Market ETF","etf",   200,  195.0, 238.70,  None),
        ("BND",  "Vanguard Bond ETF",       "bond",   300,  72.0,  74.20,   None),
        ("MSFT", "Microsoft Corporation",   "equity", 60,   320.0, 415.80,  "Technology"),
        ("VNQ",  "Vanguard Real Estate ETF","etf",    150,  82.0,  88.90,   "Real Estate"),
        ("CASH", "Cash & Equivalents",      "cash",   1,    30000.0, 30000.0, None),
    ],
    # Eleanor Vasquez — conservative, income-focused
    [
        ("TLT",  "iShares 20+ Year Treasury","bond",  400,  95.0,  91.30,   None),
        ("VYM",  "Vanguard High Div Yield ETF","etf", 500,  100.0, 112.40,  None),
        ("JNJ",  "Johnson & Johnson",       "equity", 200,  155.0, 148.90,  "Healthcare"),
        ("PG",   "Procter & Gamble",        "equity", 150,  140.0, 158.20,  "Consumer Staples"),
        ("AGG",  "iShares Core Bond ETF",   "bond",   600,  98.0,  96.80,   None),
        ("CASH", "Cash & Equivalents",      "cash",   1,    250000.0, 250000.0, None),
    ],
    # Marcus Thompson — moderate
    [
        ("SPY",  "SPDR S&P 500 ETF",        "etf",   100,  420.0, 528.60,  None),
        ("QQQ",  "Invesco NASDAQ-100 ETF",  "etf",   50,   340.0, 440.20,  None),
        ("AMZN", "Amazon.com Inc.",         "equity", 40,   130.0, 185.70,  "Consumer Discretionary"),
        ("GLD",  "SPDR Gold Shares",        "etf",   80,   170.0, 210.40,  None),
        ("CASH", "Cash & Equivalents",      "cash",   1,    20000.0, 20000.0, None),
    ],
    # Priya Kapoor — aggressive, healthcare focus
    [
        ("UNH",  "UnitedHealth Group",      "equity", 80,   480.0, 520.30,  "Healthcare"),
        ("ABBV", "AbbVie Inc.",             "equity", 120,  140.0, 175.80,  "Healthcare"),
        ("GOOGL","Alphabet Inc.",           "equity", 60,   130.0, 175.40,  "Technology"),
        ("XBI",  "SPDR Biotech ETF",        "etf",   200,  75.0,  88.20,   "Healthcare"),
        ("MSFT", "Microsoft Corporation",   "equity", 50,   380.0, 415.80,  "Technology"),
        ("CASH", "Cash & Equivalents",      "cash",   1,    40000.0, 40000.0, None),
    ],
]

# ── Compliance alerts ─────────────────────────────────────────────────────────

ALERTS = [
    # client_index, alert_type, severity, title, description
    (0, "concentration", "high",
     "High Tech Concentration — Sarah Mitchell",
     "Technology sector represents >65% of portfolio. Consider diversifying into other sectors to reduce concentration risk."),
    (0, "rebalance", "medium",
     "Annual Rebalance Due",
     "Portfolio has drifted >10% from target allocation. Rebalancing recommended."),
    (2, "risk_mismatch", "critical",
     "Bond Holding Declining — Eleanor Vasquez",
     "TLT (long-duration treasury) is down 4% this year. For a conservative profile, consider shorter-duration alternatives."),
    (3, "rebalance", "low",
     "Gold Allocation Above Target",
     "GLD now represents 18% of portfolio vs. 10% target. Consider trimming."),
    (4, "concentration", "medium",
     "Healthcare Overweight — Priya Kapoor",
     "Healthcare sector represents 55% of portfolio. Although aligned with client interest, review risk tolerance."),
]


# ── Seeder functions ──────────────────────────────────────────────────────────

def _already_seeded(conn) -> bool:
    """Check if data already exists to avoid duplicate seeding."""
    count = conn.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
    return count > 0


def seed_all(force: bool = False) -> None:
    """
    Seed the database with mock data.

    Args:
        force: If True, clears existing data and re-seeds.
    """
    init_database()

    with get_db_connection() as conn:
        if _already_seeded(conn) and not force:
            logger.info("Database already seeded. Use force=True to reseed.")
            return

        if force:
            logger.warning("Force-reseeding: clearing existing data.")
            conn.executescript("""
                DELETE FROM compliance_alerts;
                DELETE FROM chat_history;
                DELETE FROM holdings;
                DELETE FROM portfolios;
                DELETE FROM clients;
            """)

        logger.info("Seeding clients...")
        client_ids = []
        for client in CLIENTS:
            cursor = conn.execute(
                """INSERT INTO clients (name, email, phone, risk_profile, aum, advisor_notes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (client["name"], client["email"], client["phone"],
                 client["risk_profile"], client["aum"], client["advisor_notes"]),
            )
            client_ids.append(cursor.lastrowid)

        logger.info("Seeding portfolios and holdings...")
        for idx, client_id in enumerate(client_ids):
            # Create portfolio
            port_cursor = conn.execute(
                "INSERT INTO portfolios (client_id, name) VALUES (?, ?)",
                (client_id, "Main Portfolio"),
            )
            portfolio_id = port_cursor.lastrowid

            # Compute total value and insert holdings
            total_value = 0.0
            for (ticker, name, asset_class, qty, avg_cost, price, sector) in HOLDINGS_PER_CLIENT[idx]:
                conn.execute(
                    """INSERT INTO holdings
                       (portfolio_id, ticker, asset_name, asset_class, quantity, avg_cost, current_price, sector)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (portfolio_id, ticker, name, asset_class, qty, avg_cost, price, sector),
                )
                total_value += qty * price

            # Update portfolio total value
            conn.execute(
                "UPDATE portfolios SET total_value=? WHERE id=?",
                (total_value, portfolio_id),
            )
            # Update client AUM to match
            conn.execute(
                "UPDATE clients SET aum=? WHERE id=?",
                (total_value, client_id),
            )

        logger.info("Seeding compliance alerts...")
        for (client_idx, alert_type, severity, title, description) in ALERTS:
            conn.execute(
                """INSERT INTO compliance_alerts (client_id, alert_type, severity, title, description)
                   VALUES (?, ?, ?, ?, ?)""",
                (client_ids[client_idx], alert_type, severity, title, description),
            )

    logger.info("Seeding complete. %d clients loaded.", len(CLIENTS))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed the Advisor AI database.")
    parser.add_argument("--force", action="store_true", help="Clear and reseed all data.")
    args = parser.parse_args()
    seed_all(force=args.force)
    print("Done.")
