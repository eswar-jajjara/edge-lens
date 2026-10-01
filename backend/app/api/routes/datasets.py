from fastapi import APIRouter, Request, HTTPException
from app.services.datasets import ingest_dataset

router = APIRouter()
MAX_UPLOAD = 50 * 1024 * 1024


def public_dataset(dataset):
    return {key: value for key, value in dataset.items() if key not in {"path", "entries"}}


@router.post("/datasets", status_code=201)
async def upload_dataset(request: Request):
    if request.headers.get("content-type", "").split(";", 1)[0] not in {"application/zip", "application/octet-stream"}:
        raise HTTPException(415, "Upload a ZIP archive with Content-Type: application/zip.")
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_UPLOAD:
            raise HTTPException(413, "Dataset ZIP must be at most 50 MB.")
        content.extend(chunk)
    try:
        dataset = ingest_dataset(bytes(content), request.headers.get("x-dataset-name", "Image dataset")[:120], request.app.state.config.data_dir)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    request.app.state.repository.save_dataset(dataset)
    return public_dataset(dataset)


@router.get("/datasets")
def list_datasets(request: Request):
    return [public_dataset(item) for item in request.app.state.repository.list_datasets()]
