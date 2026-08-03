"""Dataset API routes for SaaS."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.auth_service import AuthContext, get_auth_context
from app.services.dataset_service import DatasetService

router = APIRouter()


@router.post("")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(None),
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Upload CSV dataset."""
    service = DatasetService(db, auth.org_id)
    dataset = await service.upload_csv(file, name, description)
    
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "row_count": dataset.row_count,
        "columns": dataset.columns,
        "uploaded_at": dataset.uploaded_at.isoformat(),
    }


@router.get("")
async def list_datasets(
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """List all datasets."""
    service = DatasetService(db, auth.org_id)
    datasets = service.list_datasets()
    
    return [
        {
            "id": d.id,
            "name": d.name,
            "description": d.description,
            "row_count": d.row_count,
            "column_count": len(d.columns),
            "file_size": d.file_size,
            "uploaded_at": d.uploaded_at.isoformat(),
        }
        for d in datasets
    ]


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Get dataset details with columns."""
    service = DatasetService(db, auth.org_id)
    dataset = service.get_dataset(dataset_id)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "row_count": dataset.row_count,
        "columns": dataset.columns,
        "file_size": dataset.file_size,
        "uploaded_at": dataset.uploaded_at.isoformat(),
    }


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    rows: int = 10,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Get preview of dataset rows."""
    service = DatasetService(db, auth.org_id)
    return service.get_dataset_preview(dataset_id, rows)


@router.get("/{dataset_id}/columns/{column}")
async def get_column_stats(
    dataset_id: str,
    column: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Get detailed statistics for a column."""
    service = DatasetService(db, auth.org_id)
    return service.get_column_stats(dataset_id, column)


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db)
):
    """Delete a dataset."""
    if not auth.has_scope("admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    service = DatasetService(db, auth.org_id)
    if not service.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {"message": "Dataset deleted"}
