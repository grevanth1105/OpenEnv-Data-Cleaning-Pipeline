FROM python:3.11-slim

# Metadata
LABEL name="data-cleaning-env" \
      version="1.0.0" \
      description="OpenEnv Data Cleaning Pipeline — RL environment for dataset cleaning tasks"

# ---------------------------------------------------------------------------
# System deps
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies — install before copying code (layer cache)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY models.py             .
COPY dataset_generator.py  .
COPY graders.py            .
COPY code_sandbox.py       .
COPY environment.py        .
COPY client.py             .
COPY baseline.py           .
COPY openenv.yaml          .
COPY server/app.py         ./server/app.py

# ---------------------------------------------------------------------------
# Environment variables (overridable at runtime)
# ---------------------------------------------------------------------------
ENV HOST=0.0.0.0 \
    PORT=7860 \
    WORKERS=4 \
    MAX_CONCURRENT_ENVS=100 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# HF Spaces uses port 7860 by default
EXPOSE 7860

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
CMD uvicorn server.app:app \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS