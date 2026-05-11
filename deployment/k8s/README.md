# geo-aware-mro Kubernetes Deployment

## Apply Namespace

kubectl apply -f namespace.yaml

## Deploy Redis

kubectl apply -f redis-deployment.yaml
kubectl apply -f redis-service.yaml

## Deploy API

kubectl apply -f api-deployment.yaml
kubectl apply -f api-service.yaml

## Deploy Celery Worker

kubectl apply -f celery-worker.yaml

## Check Pods

kubectl get pods

## Check Services

kubectl get svc
