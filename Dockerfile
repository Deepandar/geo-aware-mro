# =====================================================
# Stage 1 — Builder
# =====================================================

FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    curl \
    sqlite3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt .

RUN pip install --upgrade pip

RUN pip install \
    --prefix=/install \
    --no-cache-dir \
    -r requirements-lock.txt

# =====================================================
# Stage 2 — Runtime
# =====================================================

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    sqlite3 \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY . .

RUN useradd -m appuser

RUN mkdir -p /app/logs
RUN mkdir -p /app/artifacts
RUN mkdir -p /app/mlflow

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s \
CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
