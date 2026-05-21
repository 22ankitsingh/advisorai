# Advisor AI — Financial Advisory Platform

A production-ready AI-powered financial advisor assistant built with Streamlit, Google Gemini, SQLite, and ChromaDB.

---

## Architecture

```
AdvisorAI_new/
├── app.py                          # Entry point, routing, sidebar nav
├── data/
│   └── advisor_ai.db               # SQLite database (single source of truth)
├── database/
│   ├── schema.py                   # CREATE TABLE migrations (idempotent)
│   ├── connection.py               # Context-managed DB connections
│   └── repositories/
│       ├── client_repository.py    # CRUD for clients
│       ├── portfolio_repository.py # CRUD for portfolios + holdings; get_holdings_df()
│       ├── chat_repository.py      # Chat sessions + message history
│       └── audit_repository.py    # Audit log queries + stats
├── views/
│   ├── dashboard.py                # KPIs, AUM chart, risk breakdown, recent alerts
│   ├── client_management.py        # Full client CRUD UI
│   ├── portfolio_management.py     # Holdings CRUD, CSV import, live analytics
│   ├── portfolio.py                # Deep portfolio analytics (Sharpe, drawdown, risk)
│   ├── chat.py                     # Persistent AI chat with session history
│   ├── client_summary.py           # AI-generated advisor briefings (4 types)
│   ├── compliance.py               # Compliance alerts, scanning, resolve
│   ├── audit_logs.py               # Audit log viewer with filters + export
│   └── research.py                 # RAG-powered document search
├── services/
│   ├── gemini_service.py           # Gemini 2.5 Flash chat + streaming
│   ├── summary_service.py          # AI summary generation (DB-aware)
│   ├── chat_history_service.py     # Session lifecycle + DB persistence
│   └── database.py                 # Legacy DB helper (wraps connection.py)
├── compliance/
│   ├── rules_engine.py             # 10 compliance rules engine
│   ├── alert_service.py            # Alert orchestration (DB-aware, all clients)
│   └── audit_logger.py             # AuditLogger for compliance events
├── portfolio/
│   ├── mock_data.py                # Legacy demo data (NAV history, CLIENTS map)
│   ├── analytics.py                # Portfolio analytics (Sharpe, drift, allocation)
│   └── risk_engine.py              # Composite risk score + flag engine
└── utils/
    ├── client_resolver.py          # ClientRef unified model (DB-primary)
    ├── helpers.py                  # Formatting, chart theming, colour maps
    └── logger.py                   # Structured logging
```

---

## Database Schema

| Table               | Purpose                                              |
|---------------------|------------------------------------------------------|
| `clients`           | Client profiles (name, email, risk profile, AUM)    |
| `portfolios`        | Portfolio containers per client                      |
| `holdings`          | Individual positions (qty, price, sector, class)     |
| `transactions`      | Full transaction ledger (buy/sell/deposit/withdraw)  |
| `compliance_alerts` | Rule violations with severity + resolve workflow     |
| `chat_sessions`     | Chat session metadata (title, client FK)             |
| `chat_history`      | Individual messages (role, content, timestamps)      |
| `audit_log`         | Immutable event log (client CRUD, portfolio, chat)   |

---

## Navigation

| Page             | Description                                              |
|------------------|----------------------------------------------------------|
| 🏠 Dashboard     | AUM KPIs, client table, alert summary, charts            |
| 👥 Clients       | Full client CRUD — create, edit, delete, search          |
| 📁 Portfolio Mgmt| Holdings CRUD, CSV bulk import, transaction log          |
| 💬 AI Chat       | Persistent chat with Gemini; full session history panel  |
| 📊 Portfolio     | Deep analytics — Sharpe, drawdown, allocation drift      |
| 📋 Client Summary| 4-type AI briefing generator (meeting prep, risk, etc.)  |
| 🔍 Research      | RAG document search via ChromaDB                         |
| ⚠️ Compliance    | Rule scans, alert list, resolve workflow                  |
| 📋 Audit Log     | Filterable event log with trend chart + CSV export       |

---

## Database Migration Phases

| Phase | Commit    | Description                                         |
|-------|-----------|-----------------------------------------------------|
| 1     | `823c259` | DB schema, repositories, seed data                  |
| 2     | `c07052d` | Client Management UI + DB-primary ClientRef         |
| 3     | `14a0665` | Portfolio Management UI + CRUD + CSV import          |
| 4     | `82aa9fd` | Persistent Chat History (sessions + messages in DB) |
| 5     | `b5813b2` | Audit Log Viewer UI                                  |
| 6     | `9be71e9` | Full DB integration — mock_data.py deprecated        |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
The app runs without a Gemini key — AI features degrade to template mode.

### 3. Run
```bash
streamlit run app.py
```

The SQLite database is created automatically at `data/advisor_ai.db` on first run.

---

## Key Design Decisions

- **DB is the single source of truth** — all writes go through repositories; AUM and portfolio totals are reconciled after every mutation via `_sync()`.
- **Backward compatibility** — `portfolio/mock_data.py` still provides NAV history and the `CLIENTS` map for legacy demo clients. New DB-only clients gracefully degrade (no NAV chart, even-split allocation target).
- **Audit trail everywhere** — every CRUD operation writes an immutable `audit_log` entry. The Audit Log page surfaces this with filters, charts, and CSV export.
- **Persistent chat** — each conversation is stored as a `chat_session` with linked `chat_history` rows. Sessions are auto-titled from the first message, scoped to the active client, and restorable from the history panel.
- **Graceful AI degradation** — all AI features (chat, summaries, compliance scan logging) fall back cleanly when `GEMINI_API_KEY` is absent.

---

## Tech Stack

| Layer        | Technology                              |
|--------------|-----------------------------------------|
| UI           | Streamlit                               |
| AI           | Google Gemini 2.5 Flash (via google-generativeai) |
| Database     | SQLite (via Python stdlib `sqlite3`)    |
| Vector search| ChromaDB                                |
| Analytics    | Pandas, Plotly                          |
| Embeddings   | sentence-transformers                   |
