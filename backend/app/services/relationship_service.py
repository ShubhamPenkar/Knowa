"""Service for Multi-Dataset Intelligence: relationship discovery, grain inference, and metric safety."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Dataset
from app.ml.dataset_profiler import profile_path_for_parquet
from app.ml.relationship_inferencer import (
    DatasetGrain,
    DatasetRelationship,
    discover_candidate_relationships,
    infer_dataset_grain,
    validate_relationship_join,
)
from app.services.dataset_service import DatasetService

settings = get_settings()


def grain_path_for_parquet(parquet_path: str) -> str:
    """Return file path for dataset grain sidecar."""
    if parquet_path.endswith(".parquet"):
        return parquet_path[: -len(".parquet")] + ".grain.json"
    return parquet_path + ".grain.json"


class DatasetRelationshipService:
    """Manage multi-dataset relationship discovery, grain inference, and validation."""

    def __init__(self, db: Session, org_id: str):
        self.db = db
        self.org_id = org_id
        self.dataset_service = DatasetService(db, org_id)
        self.org_dir = os.path.join(settings.data_path, "uploads", org_id)
        os.makedirs(self.org_dir, exist_ok=True)
        self.relationships_file = os.path.join(self.org_dir, "relationships.json")

    def get_dataset_grain(self, dataset_id: str, refresh: bool = False) -> DatasetGrain:
        """Infer or retrieve cached dataset grain for a registered dataset."""
        dataset = self.dataset_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        grain_file = grain_path_for_parquet(dataset.file_path)
        if not refresh and os.path.exists(grain_file):
            try:
                with open(grain_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return DatasetGrain(**data)
            except Exception:
                pass

        if not os.path.exists(dataset.file_path):
            raise HTTPException(status_code=404, detail="Dataset storage file not found")

        df = pd.read_parquet(dataset.file_path)
        dataset_meta = {
            "id": dataset.id,
            "name": dataset.name,
            "columns": dataset.columns or [],
        }

        grain = infer_dataset_grain(df, dataset_meta)

        # Cache grain sidecar
        try:
            with open(grain_file, "w", encoding="utf-8") as f:
                json.dump(grain.to_dict(), f, indent=2)
        except Exception:
            pass

        return grain

    def discover_relationships(
        self,
        dataset_ids: Optional[list[str]] = None,
        refresh: bool = False,
    ) -> list[DatasetRelationship]:
        """Discover candidate join relationships across organization datasets."""
        # Check cached relationships file if not forcing refresh
        if not refresh and os.path.exists(self.relationships_file) and dataset_ids is None:
            try:
                with open(self.relationships_file, "r", encoding="utf-8") as f:
                    cached_list = json.load(f)
                    return [DatasetRelationship(**item) for item in cached_list]
            except Exception:
                pass

        all_datasets = self.dataset_service.list_datasets()
        if dataset_ids:
            target_ids = set(dataset_ids)
            datasets_to_analyze = [d for d in all_datasets if d.id in target_ids]
            if len(datasets_to_analyze) < len(target_ids):
                found_ids = {d.id for d in datasets_to_analyze}
                missing = target_ids - found_ids
                raise HTTPException(status_code=404, detail=f"Datasets not found: {missing}")
        else:
            datasets_to_analyze = all_datasets

        if len(datasets_to_analyze) < 2:
            return []

        # Load dataframes and metadata
        loaded: list[tuple[Dataset, pd.DataFrame, dict[str, Any]]] = []
        for d in datasets_to_analyze:
            if not os.path.exists(d.file_path):
                continue
            df = pd.read_parquet(d.file_path)
            meta = {
                "id": d.id,
                "name": d.name,
                "columns": d.columns or [],
            }
            loaded.append((d, df, meta))

        discovered: list[DatasetRelationship] = []

        # Check if previous relationships had validated or rejected statuses to preserve
        previous_decisions: dict[str, dict[str, Any]] = {}
        if os.path.exists(self.relationships_file):
            try:
                with open(self.relationships_file, "r", encoding="utf-8") as f:
                    for item in json.load(f):
                        rel_id = item.get("id")
                        val_stat = item.get("validation_status")
                        if rel_id and val_stat in ("validated", "rejected"):
                            previous_decisions[rel_id] = item
            except Exception:
                pass

        # Pairwise comparison (directional: A -> B and B -> A)
        for i in range(len(loaded)):
            for j in range(len(loaded)):
                if i == j:
                    continue
                d_src, df_src, meta_src = loaded[i]
                d_tgt, df_tgt, meta_tgt = loaded[j]

                rels = discover_candidate_relationships(
                    source_df=df_src,
                    source_meta=meta_src,
                    target_df=df_tgt,
                    target_meta=meta_tgt,
                )
                for r in rels:
                    if r.id in previous_decisions:
                        prev = previous_decisions[r.id]
                        r.validation_status = prev.get("validation_status", r.validation_status)
                        r.fan_out_ratio = prev.get("fan_out_ratio", r.fan_out_ratio)
                        if prev.get("rejection_reason"):
                            r.rejection_reason = prev.get("rejection_reason")
                        if prev.get("review_reason"):
                            r.review_reason = prev.get("review_reason")
                        if prev.get("evidence"):
                            r.evidence = prev.get("evidence")
                    discovered.append(r)

        # Persist discovered relationships
        try:
            with open(self.relationships_file, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in discovered], f, indent=2)
        except Exception:
            pass

        return discovered

    def get_relationships_for_dataset(self, dataset_id: str) -> list[DatasetRelationship]:
        """Return discovered relationships where dataset is either source or target."""
        dataset = self.dataset_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Discover or load from cache
        all_rels = self.discover_relationships(refresh=False)
        return [
            r for r in all_rels
            if r.source_dataset_id == dataset_id or r.target_dataset_id == dataset_id
        ]

    def validate_relationship(
        self,
        source_dataset_id: str,
        source_column: str,
        target_dataset_id: str,
        target_column: str,
    ) -> DatasetRelationship:
        """Validate a proposed join relationship against actual data and fan-out constraints."""
        s_dataset = self.dataset_service.get_dataset(source_dataset_id)
        t_dataset = self.dataset_service.get_dataset(target_dataset_id)

        if not s_dataset or not t_dataset:
            raise HTTPException(status_code=404, detail="Source or target dataset not found")

        df_src = pd.read_parquet(s_dataset.file_path)
        df_tgt = pd.read_parquet(t_dataset.file_path)

        meta_src = {"id": s_dataset.id, "name": s_dataset.name, "columns": s_dataset.columns or []}
        meta_tgt = {"id": t_dataset.id, "name": t_dataset.name, "columns": t_dataset.columns or []}

        # Check existing candidates or generate candidate
        candidates = discover_candidate_relationships(df_src, meta_src, df_tgt, meta_tgt)
        matching_cand = next(
            (c for c in candidates if c.source_column == source_column and c.target_column == target_column),
            None,
        )

        if matching_cand is None:
            # Build ad-hoc candidate for validation if not heuristically pre-filtered
            rel_id = f"{source_dataset_id}_{source_column}__{target_dataset_id}_{target_column}"
            matching_cand = DatasetRelationship(
                id=rel_id,
                source_dataset_id=source_dataset_id,
                source_dataset_name=s_dataset.name,
                source_column=source_column,
                target_dataset_id=target_dataset_id,
                target_dataset_name=t_dataset.name,
                target_column=target_column,
                confidence_type="heuristic",
            )

        validated = validate_relationship_join(df_src, source_column, df_tgt, target_column, matching_cand)

        # Update cache if file exists
        if os.path.exists(self.relationships_file):
            try:
                with open(self.relationships_file, "r", encoding="utf-8") as f:
                    cached_list = json.load(f)
                updated_list = []
                replaced = False
                for item in cached_list:
                    if item.get("id") == validated.id:
                        updated_list.append(validated.to_dict())
                        replaced = True
                    else:
                        updated_list.append(item)
                if not replaced:
                    updated_list.append(validated.to_dict())
                with open(self.relationships_file, "w", encoding="utf-8") as f:
                    json.dump(updated_list, f, indent=2)
            except Exception:
                pass

        return validated
