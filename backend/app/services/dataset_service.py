"""Dataset management service for SaaS."""

import io
import json
import os
import uuid
from typing import Any, Optional

import pandas as pd
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Dataset
from app.ml.dataset_profiler import (
    enrich_column_metadata,
    load_profile_report,
    profile_dataframe,
    profile_path_for_parquet,
    save_profile_report,
)

settings = get_settings()


class DatasetService:
    """Handle dataset upload, storage, and analysis."""

    def __init__(self, db: Session, org_id: str):
        self.db = db
        self.org_id = org_id
        self.upload_dir = os.path.join(settings.data_path, "uploads", org_id)
        os.makedirs(self.upload_dir, exist_ok=True)

    async def upload_csv(
        self,
        file: UploadFile,
        name: str,
        description: Optional[str] = None,
    ) -> Dataset:
        filename = file.filename or ""
        if not filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        try:
            content = await file.read()
            try:
                df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
            except (UnicodeDecodeError, UnicodeError):
                df = pd.read_csv(io.BytesIO(content), encoding="latin1")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="CSV must have at least 2 columns")

        # Base column stats (UI preview)
        columns: list[dict[str, Any]] = []
        for col in df.columns:
            col_info: dict[str, Any] = {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
            }
            sample = df[col].dropna().head(5).tolist()
            col_info["sample_values"] = [str(v) for v in sample]

            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                col_info["detected_type"] = "numeric"
                col_info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                col_info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
            elif df[col].nunique() <= 20:
                col_info["detected_type"] = "categorical"
                col_info["categories"] = [str(v) for v in df[col].dropna().unique().tolist()[:20]]
            else:
                col_info["detected_type"] = "text"
            columns.append(col_info)

        # Deterministic hygiene profile (every upload)
        report = profile_dataframe(df)
        columns = enrich_column_metadata(columns, report)

        file_id = str(uuid.uuid4())
        file_path = os.path.join(self.upload_dir, f"{file_id}.parquet")
        df.to_parquet(file_path, index=False)
        save_profile_report(report, profile_path_for_parquet(file_path))

        dataset = Dataset(
            organization_id=self.org_id,
            name=name,
            description=description,
            file_path=file_path,
            file_size=len(content),
            row_count=len(df),
            columns=columns,
        )
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        return dataset

    def list_datasets(self) -> list[Dataset]:
        return (
            self.db.query(Dataset)
            .filter(Dataset.organization_id == self.org_id)
            .order_by(Dataset.uploaded_at.desc())
            .all()
        )

    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        return (
            self.db.query(Dataset)
            .filter(Dataset.id == dataset_id, Dataset.organization_id == self.org_id)
            .first()
        )

    def get_profile(self, dataset_id: str) -> Optional[dict[str, Any]]:
        """Return saved profiling report (recompute if sidecar missing, preserving user overrides)."""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None
        path = profile_path_for_parquet(dataset.file_path)
        cached = load_profile_report(path)
        if cached is not None:
            return cached
        # Reconstruct overrides from dataset.columns if sidecar was missing
        overrides: dict[str, dict[str, Any]] = {}
        for col in dataset.columns or []:
            if col.get("is_user_corrected"):
                overrides[col["name"]] = {
                    "semantic_type": col.get("semantic_type"),
                    "role": col.get("role"),
                    "notes": "Preserved user correction",
                }
        df = pd.read_parquet(dataset.file_path)
        report = profile_dataframe(df, user_overrides=overrides)
        save_profile_report(report, path)
        return report.to_dict()

    def get_dataset_preview(self, dataset_id: str, rows: int = 10) -> dict[str, Any]:
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        df = pd.read_parquet(dataset.file_path)
        preview = df.head(rows).to_dict(orient="records")
        return {
            "columns": [col["name"] for col in dataset.columns],
            "rows": preview,
            "total_rows": dataset.row_count,
        }

    def get_column_stats(self, dataset_id: str, column: str) -> dict[str, Any]:
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        df = pd.read_parquet(dataset.file_path)
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found")

        col = df[column]
        stats: dict[str, Any] = {
            "name": column,
            "dtype": str(col.dtype),
            "count": int(col.count()),
            "null_count": int(col.isnull().sum()),
            "unique_count": int(col.nunique()),
        }
        # Attach profiler slice when available
        for meta in dataset.columns or []:
            if meta.get("name") == column:
                for k in (
                    "inferred_type",
                    "action_taken",
                    "issues",
                    "null_frac",
                    "numeric_parse_frac",
                    "profile_reason",
                    "semantic_type",
                    "role",
                    "confidence",
                    "evidence",
                    "is_user_corrected",
                ):
                    if k in meta:
                        stats[k] = meta[k]
                break

        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            stats["mean"] = float(col.mean())
            stats["std"] = float(col.std()) if col.count() else None
            stats["min"] = float(col.min())
            stats["max"] = float(col.max())
            stats["quartiles"] = {
                "25%": float(col.quantile(0.25)),
                "50%": float(col.quantile(0.50)),
                "75%": float(col.quantile(0.75)),
            }
        else:
            value_counts = col.value_counts().head(20).to_dict()
            stats["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
        return stats

    def delete_dataset(self, dataset_id: str) -> bool:
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return False
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        profile_path = profile_path_for_parquet(dataset.file_path)
        if os.path.exists(profile_path):
            os.remove(profile_path)
        self.db.delete(dataset)
        self.db.commit()
        return True

    def load_dataframe(self, dataset_id: str) -> pd.DataFrame:
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return pd.read_parquet(dataset.file_path)

    def update_column_semantic(
        self,
        dataset_id: str,
        column_name: str,
        semantic_type: str,
        role: str,
        notes: Optional[str] = None,
    ) -> Dataset:
        """Update a column's semantic type and role without modifying raw data."""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        found = False
        updated_columns = []
        user_overrides: dict[str, dict[str, Any]] = {}

        for col in dataset.columns or []:
            col_entry = dict(col)
            if col_entry.get("name") == column_name:
                col_entry["semantic_type"] = semantic_type
                col_entry["role"] = role
                col_entry["confidence"] = 1.0
                col_entry["confidence_type"] = "user_assigned"
                col_entry["is_user_corrected"] = True
                evidence_notes = [f"User corrected: {notes}"] if notes else [f"User corrected: role='{role}', semantic_type='{semantic_type}'"]
                col_entry["evidence"] = evidence_notes
                found = True
            updated_columns.append(col_entry)
            if col_entry.get("is_user_corrected"):
                user_overrides[col_entry["name"]] = {
                    "semantic_type": col_entry.get("semantic_type"),
                    "role": col_entry.get("role"),
                    "notes": notes or "User correction",
                }

        if not found:
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' not found in dataset")

        dataset.columns = updated_columns
        self.db.commit()
        self.db.refresh(dataset)

        # Update profile sidecar report with user override
        path = profile_path_for_parquet(dataset.file_path)
        cached = load_profile_report(path)
        if cached is not None:
            cached["user_overrides"] = user_overrides
            for c in cached.get("columns", []):
                if c.get("column_name") == column_name:
                    c["semantic_type"] = semantic_type
                    c["role"] = role
                    c["confidence"] = 1.0
                    c["confidence_type"] = "user_assigned"
                    c["is_user_corrected"] = True
                    c["evidence"] = [f"User corrected: {notes}"] if notes else [f"User corrected: role='{role}', semantic_type='{semantic_type}'"]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cached, f, indent=2)

        return dataset

    def batch_update_semantics(
        self,
        dataset_id: str,
        corrections: dict[str, dict[str, Any]],
    ) -> Dataset:
        """Batch update semantic annotations across multiple columns."""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        col_names = {c.get("name") for c in dataset.columns or []}
        for target_col in corrections.keys():
            if target_col not in col_names:
                raise HTTPException(status_code=400, detail=f"Column '{target_col}' not found in dataset")

        updated_columns = []
        user_overrides: dict[str, dict[str, Any]] = {}
        for col in dataset.columns or []:
            col_entry = dict(col)
            name = col_entry.get("name")
            if name in corrections:
                corr = corrections[name]
                sem_type = corr.get("semantic_type") or col_entry.get("semantic_type")
                role = corr.get("role") or col_entry.get("role")
                notes = corr.get("notes")
                col_entry["semantic_type"] = sem_type
                col_entry["role"] = role
                col_entry["confidence"] = 1.0
                col_entry["confidence_type"] = "user_assigned"
                col_entry["is_user_corrected"] = True
                col_entry["evidence"] = [f"User corrected: {notes}"] if notes else [f"User corrected: role='{role}', semantic_type='{sem_type}'"]
            updated_columns.append(col_entry)
            if col_entry.get("is_user_corrected"):
                user_overrides[col_entry["name"]] = {
                    "semantic_type": col_entry.get("semantic_type"),
                    "role": col_entry.get("role"),
                }

        dataset.columns = updated_columns
        self.db.commit()
        self.db.refresh(dataset)

        # Update profile sidecar report
        path = profile_path_for_parquet(dataset.file_path)
        cached = load_profile_report(path)
        if cached is not None:
            cached["user_overrides"] = user_overrides
            for c in cached.get("columns", []):
                name = c.get("column_name")
                if name in corrections:
                    corr = corrections[name]
                    c["semantic_type"] = corr.get("semantic_type") or c.get("semantic_type")
                    c["role"] = corr.get("role") or c.get("role")
                    c["confidence"] = 1.0
                    c["confidence_type"] = "user_assigned"
                    c["is_user_corrected"] = True
                    notes = corr.get("notes")
                    c["evidence"] = [f"User corrected: {notes}"] if notes else [f"User corrected: role='{c['role']}', semantic_type='{c['semantic_type']}'"]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cached, f, indent=2)

        return dataset
