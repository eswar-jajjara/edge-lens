from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.api.router import router
from app.core.config import Settings
from app.repositories import Repository
from app.services.validation import ValidationService
from app.services.impulse_connections import ImpulseConnections


def create_app(settings: Settings | None = None, *, runner=None, capability_provider=None) -> FastAPI:
    config = settings or Settings.from_environment()
    from app.services.benchmark import run_benchmark, runtime_capabilities

    @asynccontextmanager
    async def lifespan(application):
        config.data_dir.mkdir(parents=True, exist_ok=True)
        repository = Repository(config.data_dir)
        repository.recover_interrupted()
        repository.recover_provider_submissions()
        application.state.repository = repository
        application.state.validation = ValidationService(repository, config.data_dir, runner or run_benchmark)
        application.state.impulse_connections = ImpulseConnections()
        try:
            yield
        finally:
            application.state.impulse_connections.close()
            application.state.validation.close()

    application = FastAPI(
        title="EdgeLens API", version="0.10.0", lifespan=lifespan,
        description="Image classification conversion benchmarks and persistent layer reports.",
        docs_url="/api/docs" if config.environment == "development" else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if config.environment == "development" else None,
    )
    application.state.config = config

    @application.exception_handler(RequestValidationError)
    async def safe_validation_error(_request, exc):
        # Pydantic's default response includes the submitted value; never echo
        # malformed API keys or credentials back into frontend error messages.
        errors = [{"loc": e["loc"], "type": e["type"], "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})
    application.state.capability_provider = capability_provider or runtime_capabilities
    if config.cors_origins:
        application.add_middleware(
            CORSMiddleware, allow_origins=list(config.cors_origins), allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-Dataset-Name", "X-Model-Spec"],
        )
    application.include_router(router, prefix="/api/v1")
    return application


app = create_app()
