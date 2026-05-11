#!/usr/bin/env bash

set -e

kubectl apply --dry-run=client \
-f deployment/k8s/api-deployment.yaml

kubectl apply --dry-run=client \
-f deployment/k8s/api-service.yaml

kubectl apply --dry-run=client \
-f deployment/k8s/redis-deployment.yaml

kubectl apply --dry-run=client \
-f deployment/k8s/redis-service.yaml

kubectl apply --dry-run=client \
-f deployment/k8s/celery-worker.yaml

echo ""
echo "Kubernetes manifests validated"
