"""Dynamic Business Intelligence Service for Knowa.

Orchestrates:
1. Analysis opportunity discovery and relevance scoring.
2. Safe, grain-aware deterministic execution of descriptive, trend, comparative,
   correlation, anomaly, segmentation, and cross-dataset analyses.
3. Proactive insight synthesis, deduplication, and ranking.
4. Explainable insight inspection and verification.
5. Caching and sidecar persistence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional
from fastapi import HTTPException
import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import Dataset
from app.ml.dynamic_intelligence import (
    AnalysisOpportunityEngine,
    AnalysisExecutionEngine,
    DynamicInsightEngine,
)
from app.services.dataset_service import DatasetService
from app.services.relationship_service import DatasetRelationshipService


class DynamicIntelligenceService:
    """Multi-tenant service driving Dynamic Business Intelligence for datasets."""

    def __init__(self, db: Session, org_id: str):
        self.db = db
        self.org_id = org_id
        self.dataset_service = DatasetService(db, org_id)
        self.rel_service = DatasetRelationshipService(db, org_id)

    def _cache_path(self, dataset: Dataset) -> str:
        return dataset.file_path.replace(".parquet", ".dynamic_intelligence.json")

    def _opportunities_cache_path(self, dataset: Dataset) -> str:
        return dataset.file_path.replace(".parquet", ".dynamic_opportunities.json")

    def get_opportunities(
        self,
        dataset_id: str,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Discover and rank candidate analysis opportunities for a dataset."""
        dataset = self.dataset_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        cache_path = self._opportunities_cache_path(dataset)
        if not refresh and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    if isinstance(cached, dict) and "opportunities" in cached:
                        return cached
            except Exception:
                pass

        # Fetch semantic metadata and relationships
        semantic_profile = self.dataset_service.get_semantic_profile(dataset_id)
        grain_info = self.rel_service.get_dataset_grain(dataset_id).to_dict()
        relationships = [r.to_dict() for r in self.rel_service.discover_relationships()]

        # Generate opportunities
        opps = AnalysisOpportunityEngine.discover_opportunities(
            dataset_meta={
                "id": dataset.id,
                "name": dataset.name,
                "columns": semantic_profile.get("columns", dataset.columns or []),
                "row_count": dataset.row_count,
            },
            profile=semantic_profile,
            grain_info=grain_info,
            relationships=relationships,
        )

        result = {
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "total_opportunities": len(opps),
            "opportunities": opps,
        }

        # Cache opportunities
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

        return result

    def run_analysis(
        self,
        dataset_id: str,
        opportunity_ids: Optional[List[str]] = None,
        top_n: int = 5,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Execute selected (or top-ranked) analysis opportunities and generate insights."""
        dataset = self.dataset_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        cache_path = self._cache_path(dataset)
        if not refresh and not opportunity_ids and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    if isinstance(cached, dict) and "top_insights" in cached:
                        return cached
            except Exception:
                pass

        # Load Parquet dataframe
        if not os.path.exists(dataset.file_path):
            raise HTTPException(status_code=404, detail="Dataset parquet file not found")

        try:
            df = pd.read_parquet(dataset.file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read dataset parquet: {str(e)}")

        # Fetch opportunities
        opps_data = self.get_opportunities(dataset_id, refresh=refresh)
        all_opps = opps_data.get("opportunities", [])

        # Filter opportunities to execute
        if opportunity_ids:
            target_opps = [o for o in all_opps if o["id"] in opportunity_ids]
        else:
            target_opps = all_opps[:top_n]

        # Relationships cache for cross-dataset execution
        all_rels = [r.to_dict() for r in self.rel_service.discover_relationships()]
        secondary_df_cache: Dict[str, pd.DataFrame] = {}

        executed_analyses = []
        raw_insights = []

        for opp in target_opps:
            a_type = opp.get("analysis_type")
            if a_type == "cross_dataset":
                # Find matching relationship
                matching_rel = None
                for r in all_rels:
                    if (r["source_dataset_id"] == dataset_id or r["target_dataset_id"] == dataset_id):
                        matching_rel = r
                        break

                sec_df = None
                if matching_rel:
                    other_id = (
                        matching_rel["target_dataset_id"]
                        if matching_rel["source_dataset_id"] == dataset_id
                        else matching_rel["source_dataset_id"]
                    )
                    if other_id not in secondary_df_cache:
                        sec_ds = self.dataset_service.get_dataset(other_id)
                        if sec_ds and os.path.exists(sec_ds.file_path):
                            secondary_df_cache[other_id] = pd.read_parquet(sec_ds.file_path)
                    sec_df = secondary_df_cache.get(other_id)

                res = AnalysisExecutionEngine.execute_analysis(
                    opp, df, secondary_df=sec_df, relationship=matching_rel
                )
            else:
                res = AnalysisExecutionEngine.execute_analysis(opp, df)

            executed_analyses.append(res)
            # Generate candidate insights from each completed analysis
            ins_list = DynamicInsightEngine.generate_insights_from_analysis(res)
            raw_insights.extend(ins_list)

        # Deduplicate and rank insights
        deduped = DynamicInsightEngine.deduplicate_insights(raw_insights)
        ranked_top = DynamicInsightEngine.rank_insights(deduped, top_n=top_n)

        result = {
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "opportunities_count": len(all_opps),
            "analyses_executed_count": len(executed_analyses),
            "top_insights": ranked_top,
            "all_insights": deduped,
            "executed_analyses": executed_analyses,
            "opportunities": all_opps,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Cache result
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

        return result

    def get_insights(
        self,
        dataset_id: str,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Return proactive ranked business insights for a dataset."""
        return self.run_analysis(dataset_id=dataset_id, refresh=refresh)

    def get_insight_detail(
        self,
        dataset_id: str,
        insight_id: str,
    ) -> Dict[str, Any]:
        """Return complete explainability trace and evidence breakdown for an insight."""
        intel = self.get_insights(dataset_id=dataset_id)
        all_insights = intel.get("all_insights", [])
        insight = next((ins for ins in all_insights if ins.get("id") == insight_id), None)

        if not insight:
            raise HTTPException(status_code=404, detail="Insight not found")

        # Derive calculation formula description based on insight category
        cat = insight.get("category")
        if cat == "concentration":
            calc = "Calculated as cumulative volume of top 20% records divided by total sum"
        elif cat == "trend":
            calc = "Calculated as percentage difference between earliest and latest period buckets"
        elif cat == "comparative":
            calc = "Calculated as category aggregate relative to cohort median and baseline"
        elif cat == "correlation":
            calc = "Pearson product-moment correlation coefficient r and Student's t-distribution p-value"
        elif cat == "anomaly":
            calc = "Calculated as deviation z-score = (observed - mean) / standard_deviation against empirical baseline"
        else:
            calc = "Statistical aggregation across non-null dataset records"

        return {
            "insight": insight,
            "finding": insight.get("title", ""),
            "metric_calculation": calc,
            "data_used": f"Dataset '{intel.get('dataset_name')}' (verified non-null records)",
            "relevant_dimensions": [insight.get("affected_segment")] if insight.get("affected_segment") else [],
            "statistical_evidence": insight.get("evidence", []),
            "confidence": insight.get("confidence", 0.85),
            "limitations": insight.get("limitations", []),
        }
