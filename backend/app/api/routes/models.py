import json
from urllib.parse import unquote
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError
from app.services.models import ModelSpec, ingest_model, public_model

router = APIRouter()


@router.get("/models")
def list_models(request: Request):
    return [public_model(x) for x in request.app.state.repository.list_models()]


@router.post("/models", status_code=201)
async def upload_model(request: Request):
    if not request.app.state.config.allow_custom_models:
        raise HTTPException(403, "Developer model execution is disabled on this server. Use the local desktop tool.")
    header = request.headers.get("X-Model-Spec", "")
    if len(header) > 16000:
        raise HTTPException(413, "Model settings are too large")
    try:
        spec = ModelSpec.model_validate(json.loads(unquote(header)))
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, str(exc)[:1200]) from exc
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 100 * 1024 * 1024:
            raise HTTPException(413, "Model limit is 100 MiB")
        chunks.append(chunk)
    try:
        model = ingest_model(b"".join(chunks), spec, request.app.state.config.data_dir)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    request.app.state.repository.save_model(model)
    return public_model(model)
