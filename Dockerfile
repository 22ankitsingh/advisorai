# ─────────────────────────────────────────────────────────────────────────────
# Advisor AI — Dockerfile
# Multi-stage build: dependencies → app
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System deps required by sentence-transformers and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies stage ────────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── App stage ─────────────────────────────────────────────────────────────────
FROM deps AS app

# Copy only the application source (not dev files)
COPY . .

# Create data and log directories with correct permissions
RUN mkdir -p data/chroma logs assets

# Streamlit config — disable telemetry and set server options
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_THEME_BASE=dark

# App environment
ENV APP_ENV=production
ENV LOG_LEVEL=INFO

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
