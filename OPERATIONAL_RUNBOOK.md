# geo-aware-mro Operational Runbook

## Start API

uvicorn src.api.main:app --reload

## Start Redis

docker run -p 6379:6379 redis

## Start Celery Worker

celery -A src.orchestration.tasks worker --loglevel=info

## Swagger Docs

http://127.0.0.1:8000/docs

## Run Smoke Tests

python scripts/validation/api_smoke_test.py

## Build Docker

bash scripts/validation/docker_check.sh

## Validate Kubernetes

bash scripts/validation/k8s_check.sh
