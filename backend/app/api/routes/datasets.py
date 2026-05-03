"""Dataset management routes. Files land on disk; metadata lands in JSON store."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from app.api.schemas.dataset import DatasetSchema
from app.services import dataset_service


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetSchema, status_code=201)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetSchema:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        return dataset_service.save_uploaded(file.filename or "dataset", contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("", response_model=list[DatasetSchema])
def list_datasets() -> list[DatasetSchema]:
    return dataset_service.list_datasets()


@router.get("/{dataset_id}", response_model=DatasetSchema)
def get_dataset(dataset_id: str) -> DatasetSchema:
    d = dataset_service.get_dataset(dataset_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return d


@router.delete("/{dataset_id}", status_code=204, response_class=Response)
def delete_dataset(dataset_id: str) -> Response:
    if not dataset_service.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return Response(status_code=204)
