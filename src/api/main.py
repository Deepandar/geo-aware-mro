from fastapi import FastAPI

from src.api.middleware.request_logger import RequestLoggingMiddleware


from src.api.routes.health import router as health_router
from src.api.routes.root import router as root_router


from src.api.routes.rl_jobs import router as rl_router

from src.api.routes.job_status import router as job_status_router

from src.api.routes.jobs import router as jobs_router

from src.api.routes.distributed import router as distributed_router

from src.api.routes.mlflow_routes import router as mlflow_router

from src.api.routes.simulation import router as simulation_router

from src.api.routes.inference import router as inference_router

app = FastAPI(title="Geo-Aware MRO Decision Intelligence System", version="2.0")

app.include_router(root_router)
app.include_router(health_router)
app.include_router(inference_router)
app.include_router(simulation_router)
app.include_router(mlflow_router)
app.include_router(distributed_router)
app.include_router(jobs_router)
app.include_router(job_status_router)
app.include_router(rl_router)


app.add_middleware(RequestLoggingMiddleware)
