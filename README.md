# 💼 Advisor AI

> **AI-powered financial advisory assistant** — built with Streamlit, Google Gemini, and a modular Python backend.

---

## Overview

Advisor AI is a multi-phase financial intelligence platform for financial advisors. It combines:

- 🤖 **Conversational AI** — multi-turn Gemini-powered chat with client context injection
- 📊 **Portfolio Analytics** — holdings dashboard, risk scoring, NAV performance charts
- 📋 **AI Client Summaries** — auto-generated advisor briefings, risk explanations, and behavioral profiles
- ⚠️ **Compliance Engine** — 10-rule compliance scanner with audit logging
- 🔍 **RAG Research Search** — semantic search over uploaded PDF documents *(Phase 2 backend ready)*

---

## Project Structure

```
advisor_ai/
│
├── app.py                    # Streamlit entry point — routing & sidebar
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # Container orchestration
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
│
├── pages/                    # One file per UI page
│   ├── dashboard.py          # Home — KPIs, AUM chart, alert feed
│   ├── chat.py               # AI chat interface (Gemini)
│   ├── portfolio.py          # Holdings table + risk overview
│   ├── client_summary.py     # AI-generated summaries (4 types)
│   ├── research.py           # RAG document search
│   └── compliance.py         # Compliance alerts management
│
├── services/                 # Business logic services
│   ├── gemini_service.py     # Gemini SDK wrapper (streaming + history)
│   ├── summary_service.py    # AI summary prompt builder + generator
│   └── database.py           # SQLite schema + connection manager
│
├── portfolio/                # Portfolio analytics (Phase 3)
│   ├── mock_data.py          # Realistic mock client/holdings/NAV data
│   ├── analytics.py          # Pure analytics functions (no UI)
│   └── risk_engine.py        # Composite risk scoring + flag generation
│
├── compliance/               # Compliance engine (Phase 5)
│   ├── rules_engine.py       # 10 configurable compliance rules
│   ├── alert_service.py      # Alert orchestration + SQLite persistence
│   └── audit_logger.py       # Structured audit trail
│
├── rag/                      # RAG pipeline (Phase 2)
│   ├── document_loader.py    # PDF → Document extraction
│   └── text_chunker.py       # Document → overlapping Chunk splitting
│
├── utils/                    # Shared utilities
│   ├── config.py             # Environment config + validation
│   ├── logger.py             # Structured logging setup
│   ├── helpers.py            # Formatters, Plotly theme, HTML badges
│   └── client_resolver.py   # SQLite ID ↔ mock_data key bridge
│
└── data/                     # Auto-created at runtime
    ├── advisor_ai.db         # SQLite database
    └── chroma/               # ChromaDB vector store (Phase 2)
```

---

## Quick Start

### Prerequisites

- Python 3.10 or 3.11 (3.13 works on Windows)
- A free [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd AdvisorAI_new

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (macOS / Linux)
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** `sentence-transformers` and `chromadb` may require [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for compilation. If you have issues, try `pip install sentence-transformers --only-binary=:all:`.

### 3. Configure environment

```bash
# Copy the template
copy .env.example .env    # Windows
cp .env.example .env      # macOS / Linux

# Edit .env and add your Gemini key
GEMINI_API_KEY=your_actual_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running with Docker

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# Build and start
docker compose up --build

# Stop
docker compose down
```

Data is persisted via Docker volumes:
- `./data` → SQLite database + ChromaDB vectors
- `./logs` → Application logs

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required for AI)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Model identifier |
| `GEMINI_TEMPERATURE` | `0.7` | Response creativity (0.0–2.0) |
| `GEMINI_MAX_TOKENS` | `2048` | Max tokens per response |
| `APP_ENV` | `development` | `development` \| `staging` \| `production` |
| `DATABASE_PATH` | `data/advisor_ai.db` | SQLite file path |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB storage directory |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

> **AI features are optional.** Without a `GEMINI_API_KEY`, the app runs in "Template Mode" — all analytics, portfolio, and compliance pages work fully. Only Gemini chat and AI-generated summaries are disabled.

---

## Feature Phases

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Streamlit shell, DB schema, mock seed data | ✅ Complete |
| 2 | RAG pipeline: PDF loader, chunker | ✅ Backend ready |
| 3 | Portfolio analytics, risk engine, mock data | ✅ Complete |
| 4 | AI client summaries (4 types) | ✅ Complete |
| 5 | Compliance engine, audit logger | ✅ Complete |
| 6 | RAG UI + embedding service (planned) | 🔜 Planned |
| 7 | Recommendation engine (planned) | 🔜 Planned |

---

## Key Design Decisions

### Client ID Bridge
The sidebar loads clients from SQLite (integer PKs). Analytics modules use string keys (e.g. `"sarah_mitchell"`). `utils/client_resolver.py` provides a `ClientRef` dataclass that carries both identifiers, eliminating ad-hoc ID mapping in page code.

### Shared Utilities
`utils/helpers.py` provides formatters, colour palettes, and a `apply_dark_theme(fig)` function for Plotly. All pages import from here — no duplicated formatting logic.

### Compliance Rules
Each rule in `compliance/rules_engine.py` is an independent class with a single `check()` method. Thresholds live in the `THRESHOLDS` dict — change one number to affect all rules. New rules can be added with ~15 lines of code.

### Risk Scoring
Composite 0–100 score weighted: HHI concentration (30%) + volatility (30%) + portfolio beta (20%) + profile alignment (20%).

### Graceful AI Degradation
Every AI-dependent page works without a Gemini key. The `SummaryService` falls back to a template-based meeting brief using real analytics data.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit 1.35+ |
| LLM | Google Gemini 1.5 Flash |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Database | SQLite (built-in) |
| Analytics | Pandas, NumPy |
| Charts | Plotly |
| PDF | pypdf |
| Env | python-dotenv |

---

## Development

```bash
# Run with auto-reload
streamlit run app.py --server.runOnSave true

# Re-seed the database
python -m data.seed --force

# Run compliance scan for all clients
python -c "
from compliance.alert_service import AlertService
svc = AlertService()
for s in svc.run_for_all_clients():
    print(f'{s.client_name}: {s.violations} violations')
"
```

---

## License

MIT — see `LICENSE` for details.
