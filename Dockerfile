# =====================================================
# Geo-Aware MRO Decision Intelligence System
# Production Dockerfile
# =====================================================

# =====================================================
# Stage 1 — Builder
# =====================================================

FROM python:3.10-slim AS builder

# -----------------------------------------------------
# Environment
# -----------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# -----------------------------------------------------
# System Dependencies
# -----------------------------------------------------

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    curl \
    sqlite3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------
# Python Dependencies
# -----------------------------------------------------

COPY requirements-lock.txt .

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install -r requirements-lock.txt

# =====================================================
# Stage 2 — Runtime
# =====================================================

FROM python:3.10-slim

# -----------------------------------------------------
# Environment
# -----------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# -----------------------------------------------------
# Runtime System Packages
# -----------------------------------------------------

RUN apt-get update && apt-get install -y \
    sqlite3 \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------
# Copy Installed Python Packages
# -----------------------------------------------------

COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# -----------------------------------------------------
# Copy Application Source
# -----------------------------------------------------

COPY . .

# -----------------------------------------------------
# Create Non-Root User
# -----------------------------------------------------

RUN useradd -m -s /bin/bash appuser

# -----------------------------------------------------
# Runtime Directories
# -----------------------------------------------------

RUN mkdir -p \
    /app/logs \
    /app/artifacts \
    /app/mlflow

RUN chown -R appuser:appuser /app

USER appuser

# -----------------------------------------------------
# Networking
# -----------------------------------------------------

EXPOSE 8000

# -----------------------------------------------------
# Health Check
# -----------------------------------------------------

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# -----------------------------------------------------
# Application Startup
# -----------------------------------------------------

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]