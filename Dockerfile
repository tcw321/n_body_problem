# Multi-stage build: install dependencies in a builder stage so the final
# image does not contain pip caches or build tools.

# ── Stage 1: build ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-app.txt


# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY main.py visualize.py ws_server.py sim_runner.py ./
COPY static/ ./static/

EXPOSE 8000

# Default: start the web server
CMD ["uvicorn", "ws_server:app", "--host", "0.0.0.0", "--port", "8000"]
