"""Dataset API routes for SaaS."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import BatchSemanticUpdate, ColumnSemanticUpdate
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
    profile = service.get_profile(dataset.id)

    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "row_count": dataset.row_count,
        "columns": dataset.columns,
        "uploaded_at": dataset.uploaded_at.isoformat(),
        "profiling": {
            "excluded_as_id": (profile or {}).get("excluded_as_id", []),
            "dropped_as_constant": (profile or {}).get("dropped_as_constant", []),
            "warnings": (profile or {}).get("warnings", []),
            "blocking": (profile or {}).get("blocking", False),
        },
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
        "profiling": service.get_profile(dataset_id),
    }


@router.get("/{dataset_id}/profile")
async def get_dataset_profile(
    dataset_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Return deterministic hygiene profile for a dataset."""
    service = DatasetService(db, auth.org_id)
    profile = service.get_profile(dataset_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return profile


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


@router.patch("/{dataset_id}/columns/{column}/semantic")
async def update_column_semantic(
    dataset_id: str,
    column: str,
    update: ColumnSemanticUpdate,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """User correction for a column's semantic type and role."""
    service = DatasetService(db, auth.org_id)
    dataset = service.update_column_semantic(
        dataset_id=dataset_id,
        column_name=column,
        semantic_type=update.semantic_type,
        role=update.role,
        notes=update.notes,
    )
    col_meta = next((c for c in dataset.columns if c.get("name") == column), None)
    return {
        "dataset_id": dataset.id,
        "column": column,
        "updated": col_meta,
    }


@router.put("/{dataset_id}/semantics")
async def batch_update_semantics(
    dataset_id: str,
    body: BatchSemanticUpdate,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Batch update user corrections across multiple columns."""
    service = DatasetService(db, auth.org_id)
    corrections_dict = {
        col: {"semantic_type": upd.semantic_type, "role": upd.role, "notes": upd.notes}
        for col, upd in body.corrections.items()
    }
    dataset = service.batch_update_semantics(dataset_id, corrections_dict)
    return {
        "dataset_id": dataset.id,
        "columns": dataset.columns,
    }
