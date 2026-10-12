"""Product Intelligence service orchestrating multi-dataset analysis.

Synthesizes data-driven metrics, explainable business insights,
investigation opportunities, and data-safety warnings across registered datasets.
Maintains tenant isolation, limits resource consumption, and respects dataset grain.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import Dataset
from app.ml.product_intelligence import (
    compute_column_summary,
    compute_categorical_breakdown,
    compute_temporal_trend,
    compute_top_performers,
    compute_cross_dataset_metric,
    generate_explainable_insights,
    surface_investigation_opportunities,
)
from app.services.dataset_service import DatasetService
from app.services.relationship_service import DatasetRelationshipService


class ProductIntelligenceService:
    """Multi-tenant service producing the Dynamic Product Intelligence overview."""

    MAX_ROWS_SCAN = 50_000

    def __init__(self, db: Session, org_id: str):
        self.db = db
        self.org_id = org_id
        self.dataset_service = DatasetService(db, org_id)
        self.rel_service = DatasetRelationshipService(db, org_id)

    def _get_cache_path(self) -> str:
        return os.path.join(self.dataset_service.upload_dir, "product_intelligence_overview.json")

    def build_overview(
        self,
        dataset_ids: Optional[List[str]] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Build or return cached Product Intelligence overview."""
        cache_path = self._get_cache_path()

        # Check cache if refresh not requested and dataset_ids is not a subset filter
        if not refresh and not dataset_ids and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    if isinstance(cached, dict) and "readiness" in cached:
                        return cached
            except Exception:
                pass

        # 1. Fetch authorized datasets for organization
        all_datasets = (
            self.db.query(Dataset)
            .filter(Dataset.organization_id == self.org_id)
            .order_by(Dataset.uploaded_at.desc())
            .all()
        )

        if dataset_ids:
            datasets = [d for d in all_datasets if d.id in dataset_ids]
        else:
            datasets = all_datasets

        df_cache: Dict[str, Optional[pd.DataFrame]] = {}
        dataset_summaries: List[Dict[str, Any]] = []
        metrics: List[Dict[str, Any]] = []
        unavailable_metrics: List[Dict[str, Any]] = []
        data_quality_warnings: List[Dict[str, Any]] = []
        safety_warnings: List[Dict[str, Any]] = []

        total_rows = 0
        ready_datasets_count = 0

        all_columns_map: Dict[str, list] = {}
        # Scan each dataset
        for d in datasets:
            profile = self.dataset_service.get_profile(d.id)
            grain_info = self.rel_service.get_dataset_grain(d.id).to_dict()

            raw_columns = (profile.get("columns", []) if profile else []) or (d.columns or [])
            columns_meta = []
            for c in raw_columns:
                if isinstance(c, dict):
                    col_name = c.get("name") or c.get("column_name")
                    if col_name:
                        entry = dict(c)
                        entry["name"] = str(col_name)
                        columns_meta.append(entry)
                elif isinstance(c, str):
                    columns_meta.append({"name": str(c), "role": "dimension"})

            all_columns_map[d.id] = columns_meta

            meta = {
                "id": d.id,
                "name": d.name,
                "row_count": d.row_count or 0,
                "column_count": len(columns_meta),
                "columns": columns_meta,
            }

            total_rows += meta["row_count"]

            # Load dataframe once
            try:
                df = self.dataset_service.load_dataframe(d.id)
                if df is not None and len(df) > self.MAX_ROWS_SCAN:
                    df = df.head(self.MAX_ROWS_SCAN)
            except Exception:
                df = None
            df_cache[d.id] = df

            cols = meta["columns"]
            df_cols = set(df.columns) if df is not None else set()
            measure_cols = [
                c["name"] for c in cols
                if c["name"] in df_cols and (
                    c.get("role") == "measure"
                    or c.get("semantic_type") in ("numeric_measure", "currency", "percentage")
                    or (pd.api.types.is_numeric_dtype(df[c["name"]]) and c.get("role") != "identifier" and c.get("semantic_type") not in ("id", "id_numeric", "boolean"))
                )
            ]
            dim_cols = [
                c["name"] for c in cols
                if c["name"] in df_cols and (
                    c.get("role") in ("dimension", "categorical")
                    or c.get("semantic_type") in ("categorical", "geographic", "boolean")
                    or (str(df[c["name"]].dtype) in ("object", "category") and c.get("role") != "identifier")
                )
            ]
            time_cols = [
                c["name"] for c in cols
                if c["name"] in df_cols and (c.get("role") == "timestamp" or c.get("semantic_type") == "datetime")
            ]
            id_cols = [
                c["name"] for c in cols
                if c["name"] in df_cols and (
                    c.get("role") == "identifier"
                    or c.get("semantic_type") in ("id", "id_numeric", "id_uuid")
                    or "id" in c["name"].lower()
                )
            ]

            # Readiness classification
            if df is None or df.empty:
                readiness_status = "empty"
                notes = ["Dataset contains zero loaded records"]
            elif not measure_cols and not dim_cols:
                readiness_status = "partial"
                notes = ["No dimensions or measures classified in dataset profile"]
            else:
                readiness_status = "ready"
                ready_datasets_count += 1
                notes = [f"{len(measure_cols)} measures, {len(dim_cols)} dimensions ready for analysis"]

            dataset_summaries.append({
                "dataset_id": d.id,
                "dataset_name": d.name,
                "row_count": meta["row_count"],
                "column_count": meta["column_count"],
                "grain_type": grain_info.get("grain_type", "unknown"),
                "primary_key": grain_info.get("primary_key", []),
                "measures_count": len(measure_cols),
                "dimensions_count": len(dim_cols),
                "timestamps_count": len(time_cols),
                "readiness_status": readiness_status,
                "notes": notes,
            })

            if df is None or df.empty:
                continue

            # 2. Compute Data-Driven Metrics for this dataset
            # (a) Measure Summaries & Distributions
            for m_col in measure_cols[:4]:  # Top 4 measures per dataset max
                summary = compute_column_summary(df[m_col], m_col, d.id, d.name)
                if summary:
                    metrics.append(summary)

            # (b) Categorical Breakdowns
            for dim_col in dim_cols[:3]:  # Top 3 dimensions
                # Pair with first measure if available, else frequency
                pair_measure = measure_cols[0] if measure_cols else None
                breakdown = compute_categorical_breakdown(df, dim_col, pair_measure, d.id, d.name)
                if breakdown:
                    metrics.append(breakdown)

            # (c) Temporal Trends
            if time_cols:
                time_col = time_cols[0]
                pair_measure = measure_cols[0] if measure_cols else None
                trend = compute_temporal_trend(df, time_col, pair_measure, d.id, d.name)
                if trend:
                    metrics.append(trend)

            # (d) Top Performers
            entity_candidate = (
                (id_cols[0] if id_cols else None)
                or (grain_info.get("primary_key", [None])[0] if grain_info.get("primary_key") else None)
                or (dim_cols[0] if dim_cols else None)
            )
            if entity_candidate and measure_cols and entity_candidate != measure_cols[0]:
                perf = compute_top_performers(df, entity_candidate, measure_cols[0], d.id, d.name)
                if perf:
                    metrics.append(perf)

            # (e) Data Quality Checks
            for col_info in cols:
                col_name = col_info.get("name")
                if col_name in df.columns:
                    s = df[col_name]
                    null_pct = (s.isna().sum() / len(s)) * 100.0 if len(s) else 0.0
                    if null_pct > 20.0:
                        data_quality_warnings.append({
                            "dataset_id": d.id,
                            "dataset_name": d.name,
                            "column_name": col_name,
                            "warning_type": "missingness",
                            "message": f"Column '{col_name}' has {null_pct:.1f}% missing values",
                            "impact": "Incomplete records may introduce selection bias into aggregations",
                        })
                    is_categorical_or_string = (
                        not pd.api.types.is_numeric_dtype(s)
                        and not pd.api.types.is_datetime64_any_dtype(s)
                    )
                    if is_categorical_or_string and s.nunique() > 40:
                        data_quality_warnings.append({
                            "dataset_id": d.id,
                            "dataset_name": d.name,
                            "column_name": col_name,
                            "warning_type": "high_cardinality",
                            "message": f"High cardinality dimension '{col_name}' ({s.nunique()} distinct values)",
                            "impact": "Groupings are truncated to top categories to avoid fragmented views",
                        })

        # 3. Cross-Dataset Metrics and Relationship Safety
        all_discovered_rels: list[Any] = []
        if len(datasets) > 1:
            try:
                rels = self.rel_service.discover_relationships(
                    dataset_ids=[d.id for d in datasets],
                    refresh=refresh,
                )
                all_discovered_rels = rels
                for r in rels:
                    r_dict = r.to_dict()
                    src_id = r_dict["source_dataset_id"]
                    tgt_id = r_dict["target_dataset_id"]
                    src_df = df_cache.get(src_id)
                    tgt_df = df_cache.get(tgt_id)

                    src_meta = next((s for s in dataset_summaries if s["dataset_id"] == src_id), None)
                    tgt_meta = next((s for s in dataset_summaries if s["dataset_id"] == tgt_id), None)

                    if src_df is not None and tgt_df is not None and src_meta and tgt_meta:
                        cross_metric, cross_safety = compute_cross_dataset_metric(
                            source_df=src_df,
                            source_meta={
                                "id": src_id,
                                "name": src_meta["dataset_name"],
                                "columns": all_columns_map.get(src_id, []),
                            },
                            source_grain={"grain_type": src_meta["grain_type"]},
                            target_df=tgt_df,
                            target_meta={
                                "id": tgt_id,
                                "name": tgt_meta["dataset_name"],
                                "columns": all_columns_map.get(tgt_id, []),
                            },
                            target_grain={"grain_type": tgt_meta["grain_type"]},
                            rel=r_dict,
                        )
                        if cross_metric:
                            metrics.append(cross_metric)
                        safety_warnings.extend(cross_safety)
            except Exception:
                pass

        # Deduplicate safety warnings by message and dataset pair to avoid repetitive notices
        seen_warnings = set()
        deduped_safety_warnings = []
        for sw in safety_warnings:
            key = (sw.get("source_dataset_id"), sw.get("target_dataset_id"), sw.get("warning_type"), sw.get("message"))
            if key not in seen_warnings:
                seen_warnings.add(key)
                deduped_safety_warnings.append(sw)
        safety_warnings = deduped_safety_warnings

        # 4. Check for Unavailable Analysis Capabilities
        has_timestamps = any(s["timestamps_count"] > 0 for s in dataset_summaries)
        if not has_timestamps:
            unavailable_metrics.append({
                "category": "trend",
                "label": "Temporal Trends Over Time",
                "reason": "No valid date or timestamp column identified across uploaded datasets",
            })

        has_measures = any(s["measures_count"] > 0 for s in dataset_summaries)
        if not has_measures:
            unavailable_metrics.append({
                "category": "summary",
                "label": "Measure Distributions & Financials",
                "reason": "No numeric measure or quantity columns available in uploaded datasets",
            })

        if len(datasets) > 1 and not any(m.get("category") == "cross_dataset" for m in metrics):
            if all_discovered_rels:
                has_validated = any(r.validation_status == "validated" for r in all_discovered_rels)
                if not has_validated:
                    reason = "Cross-dataset metrics are unavailable because candidate relationships require explicit review and validation before analytical use."
                else:
                    reason = "Cross-dataset joins excluded due to high fan-out row multiplication or many-to-many grain mismatch."
            else:
                reason = "No matching primary/foreign key relationships discovered between datasets"
            unavailable_metrics.append({
                "category": "cross_dataset",
                "label": "Unified Cross-Dataset Intelligence",
                "reason": reason,
            })

        # 5. Synthesize Explainable Insights & Opportunities
        insights = generate_explainable_insights(metrics)
        opportunities = surface_investigation_opportunities(metrics, insights)

        result = {
            "organization_id": self.org_id,
            "readiness": {
                "total_datasets": len(datasets),
                "ready_datasets": ready_datasets_count,
                "total_rows": total_rows,
                "dataset_summaries": dataset_summaries,
            },
            "metrics": metrics,
            "unavailable_metrics": unavailable_metrics,
            "insights": insights,
            "opportunities": opportunities,
            "data_quality_warnings": data_quality_warnings,
            "safety_warnings": safety_warnings,
            "relationships": [r.to_dict() for r in all_discovered_rels],
        }

        # Cache result if it covers all datasets
        if not dataset_ids:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
            except Exception:
                pass

        return result
