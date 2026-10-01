from fastapi import APIRouter
from app.api.routes import health, runs, datasets, models, edge

router = APIRouter()
router.include_router(models.router, tags=["Developer models"])
router.include_router(edge.router, tags=["Edge benchmarks"])
router.include_router(health.router, tags=["System"])
router.include_router(datasets.router, tags=["Datasets"])
router.include_router(runs.router, tags=["Benchmark runs"])
