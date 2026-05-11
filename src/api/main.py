from fastapi import FastAPI

from src.api.routes.health import router as health_router
from src.api.routes.root import router as root_router




from src.api.routes.distributed import (
    router as distributed_router
)

from src.api.routes.mlflow_routes import (
    router as mlflow_router
)

from src.api.routes.simulation import (
    router as simulation_router
)

from src.api.routes.inference import (
    router as inference_router
)


app = FastAPI(
    title="geo-aware-mro",
    version="2.0"
)

app.include_router(root_router)
app.include_router(health_router)
app.include_router(inference_router)
app.include_router(simulation_router)
app.include_router(mlflow_router)
app.include_router(distributed_router)
