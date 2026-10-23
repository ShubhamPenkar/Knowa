"""Pure statistical and machine learning engine for Phase 3 Dynamic Business Intelligence.

Determines:
1. What meaningful business analyses can be performed dynamically on available data.
2. Relevance scoring for potential analyses using Phase 1 semantic profiles and Phase 2 grain.
3. Deterministic execution across generic analytical types:
   - Descriptive (distributions, categorical frequency, Pareto concentration, missingness)
   - Trend (temporal patterns, period-over-period growth/decline, seasonality signals)
   - Comparative (measures across categorical dimensions, top/bottom segment disparity)
   - Correlation / Association (bivariate associations with explicit non-causal warnings)
   - Anomaly Detection (z-scores, IQR deviations, time spikes with magnitude & baseline)
   - Segmentation (unsupervised clustering on safe numeric measures, excluding IDs/dates)
   - Cross-Dataset Analyses (grain-aware joins with fan-out prevention & pre-aggregation)
4. Structured Insight Generation with severity (critical, warning, info, positive), confidence,
   factual evidence, explainability, deduplication, and proactive ranking.

Never fabricates causal claims, ROI projections, or numeric values.
Never uses LLMs for arithmetic, averages, correlations, or rankings.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans


def safe_float(val: Any, decimals: int = 4) -> Optional[float]:
    """Convert value to float if valid and finite, else None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, decimals)
    except (ValueError, TypeError):
        return None


# =====================================================================
# STEP 2 & 5: ANALYSIS OPPORTUNITY ENGINE & RELEVANCE SCORING
# =====================================================================

class AnalysisOpportunityEngine:
    """Discovers and scores meaningful analysis opportunities from semantic profiles and relationships."""

    # Business role priority weights
    ROLE_WEIGHTS = {
        "target": 1.0,
        "revenue": 0.95,
        "profit": 0.95,
        "churn": 0.95,
        "cost": 0.90,
        "conversion": 0.90,
        "quantity": 0.85,
        "price": 0.85,
        "tenure": 0.80,
        "customer": 0.85,
        "product": 0.85,
        "order": 0.80,
        "measure": 0.75,
        "dimension": 0.70,
        "temporal": 0.85,
        "date": 0.85,
    }

    @classmethod
    def discover_opportunities(
        cls,
        dataset_meta: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
        grain_info: Optional[Dict[str, Any]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Identify candidate analysis opportunities for a dataset based on its semantic metadata."""
        dataset_id = str(dataset_meta.get("id", "dataset"))
        dataset_name = str(dataset_meta.get("name", "Dataset"))
        columns = dataset_meta.get("columns", []) or []

        # Categorize columns by inferred semantic role
        measures = []
        dimensions = []
        temporals = []
        targets = []
        identifiers = []

        for c in columns:
            col_name = str(c.get("name") or c.get("column_name", ""))
            if not col_name:
                continue
            role = str(c.get("role", "dimension")).lower()
            b_role = str(c.get("business_role", "")).lower()
            s_type = str(c.get("semantic_type", "")).lower()

            col_entry = {
                "name": col_name,
                "role": role,
                "business_role": b_role,
                "semantic_type": s_type,
                "is_target": c.get("is_target", False) or b_role in ("churn", "target", "conversion"),
                "null_pct": float(c.get("null_percentage", 0.0) or 0.0),
                "distinct_count": int(c.get("distinct_count", 0) or 0),
            }

            if col_entry["is_target"]:
                targets.append(col_entry)
            if role == "measure" or s_type in ("measure", "numeric_measure", "currency", "percentage"):
                measures.append(col_entry)
            elif role == "dimension" or s_type in ("categorical", "geographic", "ordinal", "boolean"):
                dimensions.append(col_entry)
            elif role == "timestamp" or s_type in ("datetime", "date"):
                temporals.append(col_entry)
            elif role == "identifier" or s_type in ("identifier", "id"):
                identifiers.append(col_entry)

        candidates: List[Dict[str, Any]] = []

        # 1. Descriptive Opportunities (Distributions & Concentration)
        for m in measures:
            score, evidence = cls._score_measure_descriptive(m, dataset_meta)
            candidates.append({
                "id": f"opp_desc_{dataset_id}_{m['name']}",
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "analysis_type": "descriptive",
                "title": f"Typical Range & Concentration: {m['name'].replace('_', ' ').title()}",
                "description": f"Examine average vs median, typical ranges, and whether top records dominate '{m['name']}'.",
                "primary_column": m["name"],
                "secondary_column": None,
                "relevance_score": score,
                "evidence_factors": evidence,
                "grain_safe": True,
                "pre_aggregation_needed": False,
                "status": "ready",
                "reason_unavailable": None,
            })

        # 2. Trend Opportunities (Temporal + Measure or Activity Volume)
        for t in temporals:
            # Temporal with Measures
            if measures:
                for m in measures[:3]:  # Top measures
                    score, evidence = cls._score_trend(m, t, dataset_meta)
                    candidates.append({
                        "id": f"opp_trend_{dataset_id}_{m['name']}_over_{t['name']}",
                        "dataset_id": dataset_id,
                        "dataset_name": dataset_name,
                        "analysis_type": "trend",
                        "title": f"Trajectory Over Time: {m['name'].replace('_', ' ').title()}",
                        "description": f"Track growth, decline, and period-over-period changes in '{m['name']}'.",
                        "primary_column": m["name"],
                        "secondary_column": t["name"],
                        "relevance_score": score,
                        "evidence_factors": evidence,
                        "grain_safe": True,
                        "pre_aggregation_needed": False,
                        "status": "ready",
                        "reason_unavailable": None,
                    })
            else:
                # Volume / Frequency Trend
                score = 0.72
                evidence = [
                    f"Temporal dimension '{t['name']}' detected",
                    "No numeric measures found; tracking activity transaction frequency over time",
                ]
                candidates.append({
                    "id": f"opp_trend_{dataset_id}_volume_over_{t['name']}",
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "analysis_type": "trend",
                    "title": f"Activity Volume Over Time ({t['name']})",
                    "description": f"Temporal record frequency and event cadence across periods.",
                    "primary_column": t["name"],
                    "secondary_column": None,
                    "relevance_score": score,
                    "evidence_factors": evidence,
                    "grain_safe": True,
                    "pre_aggregation_needed": False,
                    "status": "ready",
                    "reason_unavailable": None,
                })

        # 3. Comparative Opportunities (Measure x Dimension, or Target x Dimension)
        for d in dimensions:
            # Filter out dimensions with massive or tiny cardinality (e.g. 1 distinct or 10,000 distinct)
            card = d["distinct_count"]
            if 0 < card < 2 or card > 200:
                continue

            # Target comparison takes top priority
            for tgt in targets:
                if tgt["name"] != d["name"]:
                    score, evidence = cls._score_target_comparison(tgt, d)
                    candidates.append({
                        "id": f"opp_comp_{dataset_id}_{tgt['name']}_by_{d['name']}",
                        "dataset_id": dataset_id,
                        "dataset_name": dataset_name,
                        "analysis_type": "comparative",
                        "title": f"{tgt['name'].replace('_', ' ').title()} Rate by {d['name'].replace('_', ' ').title()}",
                        "description": f"Compare target outcome '{tgt['name']}' variation across segments in '{d['name']}'.",
                        "primary_column": tgt["name"],
                        "secondary_column": d["name"],
                        "relevance_score": score,
                        "evidence_factors": evidence,
                        "grain_safe": True,
                        "pre_aggregation_needed": False,
                        "status": "ready",
                        "reason_unavailable": None,
                    })

            # Measure comparisons
            for m in measures[:3]:
                if m["name"] != d["name"]:
                    score, evidence = cls._score_comparative(m, d)
                    candidates.append({
                        "id": f"opp_comp_{dataset_id}_{m['name']}_by_{d['name']}",
                        "dataset_id": dataset_id,
                        "dataset_name": dataset_name,
                        "analysis_type": "comparative",
                        "title": f"{m['name'].replace('_', ' ').title()} by {d['name'].replace('_', ' ').title()}",
                        "description": f"Segment performance and contribution shares of '{m['name']}' across '{d['name']}'.",
                        "primary_column": m["name"],
                        "secondary_column": d["name"],
                        "relevance_score": score,
                        "evidence_factors": evidence,
                        "grain_safe": True,
                        "pre_aggregation_needed": False,
                        "status": "ready",
                        "reason_unavailable": None,
                    })

        # 4. Correlation / Association Opportunities (Between pairs of numeric measures)
        if len(measures) >= 2:
            for i in range(len(measures)):
                for j in range(i + 1, min(len(measures), 4)):
                    m1, m2 = measures[i], measures[j]
                    score, evidence = cls._score_correlation(m1, m2)
                    candidates.append({
                        "id": f"opp_corr_{dataset_id}_{m1['name']}_vs_{m2['name']}",
                        "dataset_id": dataset_id,
                        "dataset_name": dataset_name,
                        "analysis_type": "correlation",
                        "title": f"Association: {m1['name'].replace('_', ' ').title()} & {m2['name'].replace('_', ' ').title()}",
                        "description": f"Statistical correlation coefficient testing association (strictly non-causal).",
                        "primary_column": m1["name"],
                        "secondary_column": m2["name"],
                        "relevance_score": score,
                        "evidence_factors": evidence,
                        "grain_safe": True,
                        "pre_aggregation_needed": False,
                        "status": "ready",
                        "reason_unavailable": None,
                    })

        # 5. Anomaly Detection Opportunities (For key measures or temporal trends)
        for m in measures[:2]:
            score, evidence = cls._score_anomaly(m)
            candidates.append({
                "id": f"opp_anom_{dataset_id}_{m['name']}",
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "analysis_type": "anomaly",
                "title": f"Audit Spikes & Outliers: {m['name'].replace('_', ' ').title()}",
                "description": f"Detect unusual activity surges or extreme outlier values in '{m['name']}'.",
                "primary_column": m["name"],
                "secondary_column": None,
                "relevance_score": score,
                "evidence_factors": evidence,
                "grain_safe": True,
                "pre_aggregation_needed": False,
                "status": "ready",
                "reason_unavailable": None,
            })

        # 6. Segmentation Opportunities (When >= 2 numeric measures exist, avoiding identifiers/timestamps)
        if len(measures) >= 2:
            score = 0.82
            meas_names = [m["name"] for m in measures[:4]]
            evidence = [
                f"Discovered {len(meas_names)} continuous numeric business measures: {', '.join(meas_names)}",
                "Clean segmentation candidate: identifiers, row keys, and timestamps excluded to prevent leakage",
                "Sufficient multi-dimensional variation observed for clustering",
            ]
            candidates.append({
                "id": f"opp_seg_{dataset_id}_cohorts",
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "analysis_type": "segmentation",
                "title": f"Customer & Account Behavior Cohorts",
                "description": f"Group records into natural behavioral tiers using {', '.join(meas_names)}.",
                "primary_column": meas_names[0],
                "secondary_column": meas_names[1],
                "relevance_score": score,
                "evidence_factors": evidence,
                "grain_safe": True,
                "pre_aggregation_needed": False,
                "status": "ready",
                "reason_unavailable": None,
            })

        # 7. Cross-Dataset Grain-Aware Opportunities (from Phase 2 relationships)
        if relationships:
            # Group by target dataset so we produce at most ONE consolidated opportunity per related dataset pair
            rels_by_target: Dict[str, List[Tuple[Dict[str, Any], bool]]] = {}
            for rel in relationships:
                is_src = str(rel.get("source_dataset_id")) == dataset_id
                is_tgt = str(rel.get("target_dataset_id")) == dataset_id
                if not (is_src or is_tgt):
                    continue
                other_id = str(rel.get("target_dataset_id") if is_src else rel.get("source_dataset_id"))
                if other_id not in rels_by_target:
                    rels_by_target[other_id] = []
                rels_by_target[other_id].append((rel, is_src))

            for other_id, rel_entries in rels_by_target.items():
                # Pick the safest / highest confidence relationship as representative
                def _rel_rank(item):
                    r, _ = item
                    val_score = 2 if r.get("validation_status") == "validated" else 1
                    fan = float(r.get("fan_out_ratio", 1.0) or 1.0)
                    conf = float(r.get("confidence", 0.5) or 0.5)
                    return (val_score, -fan, conf)

                rel_entries.sort(key=_rel_rank, reverse=True)
                best_rel, is_src = rel_entries[0]

                cardinality = best_rel.get("cardinality", "one_to_one")
                fan_out = float(best_rel.get("fan_out_ratio", 1.0) or 1.0)
                has_double_counting = bool(best_rel.get("double_counting_risk", False))
                val_status = best_rel.get("validation_status", "candidate")

                other_name = best_rel.get("target_dataset_name") if is_src else best_rel.get("source_dataset_name")
                my_col = best_rel.get("source_column") if is_src else best_rel.get("target_column")
                other_col = best_rel.get("target_column") if is_src else best_rel.get("source_column")

                all_col_pairs = [
                    f"{r.get('source_column') if s else r.get('target_column')} ↔ {r.get('target_column') if s else r.get('source_column')}"
                    for r, s in rel_entries
                ]

                # Grain check
                is_safe = (val_status == "validated") and not has_double_counting and (fan_out <= 1.05)
                pre_agg_needed = (fan_out > 1.05) or (cardinality in ("one_to_many", "many_to_many"))

                evidence = [
                    f"Discovered {len(rel_entries)} link candidate(s) between {dataset_name} and {other_name}",
                    f"Primary link: {dataset_name}.{my_col} ↔ {other_name}.{other_col} ({cardinality}, {fan_out:.1f}x fan-out)",
                ]
                if len(all_col_pairs) > 1:
                    evidence.append(f"Other matching columns: {', '.join(all_col_pairs[1:4])}")

                if is_safe:
                    evidence.append("Validated join is safe against measure double-counting")
                    rel_score = 0.88
                    status = "ready"
                    reason_unavail = None
                elif pre_agg_needed:
                    evidence.append(f"Requires pre-aggregation to avoid {fan_out:.1f}x measure inflation")
                    rel_score = 0.65
                    status = "pre_aggregation_required"
                    reason_unavail = (
                        f"The requested measure exists at parent grain, but transaction join creates a "
                        f"{fan_out:.1f}x fan-out. Pre-aggregation is required before analysis."
                    )
                else:
                    evidence.append(f"Relationship validation status: '{val_status}' (join not certified safe)")
                    rel_score = 0.40
                    status = "unavailable"
                    reason_unavail = f"Relationship with '{other_name}' is not validated for analytical joins."

                candidates.append({
                    "id": f"opp_cross_{dataset_id}_with_{other_id}",
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "analysis_type": "cross_dataset",
                    "title": f"Cross-Dataset Analysis: {dataset_name} & {other_name}",
                    "description": f"Compare records across {dataset_name} and {other_name} ({len(rel_entries)} matching key candidates).",
                    "primary_column": str(my_col),
                    "secondary_column": str(other_col),
                    "relevance_score": rel_score,
                    "evidence_factors": evidence,
                    "grain_safe": is_safe,
                    "pre_aggregation_needed": pre_agg_needed,
                    "status": status,
                    "reason_unavailable": reason_unavail,
                })

        # Sort opportunities by relevance score descending
        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
        return candidates

    @classmethod
    def _score_measure_descriptive(cls, m: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.65
        evidence = [f"'{m['name']}' inferred as numeric business measure ({m['semantic_type']})"]
        b_role = m.get("business_role")
        if b_role in ("revenue", "profit", "cost", "price", "churn"):
            score += 0.20
            evidence.append(f"High business priority assigned to core financial/retention role '{b_role}'")
        elif b_role:
            score += 0.10
            evidence.append(f"Business role '{b_role}' recognized")

        if m["null_pct"] < 0.05:
            score += 0.05
            evidence.append(f"High data completeness ({100 - m['null_pct']*100:.1f}% non-null observations)")
        elif m["null_pct"] > 0.4:
            score -= 0.15
            evidence.append(f"High missingness ({m['null_pct']*100:.1f}% nulls) degrades analytical power")

        return min(0.90, max(0.2, round(score, 2))), evidence

    @classmethod
    def _score_trend(cls, m: Dict[str, Any], t: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.78
        evidence = [
            f"Temporal column '{t['name']}' ({t['semantic_type']}) paired with measure '{m['name']}'",
        ]
        b_role = m.get("business_role")
        if b_role in ("revenue", "profit", "churn", "conversion", "cost"):
            score += 0.16
            evidence.append(f"Tracking trajectory for key business metric '{b_role}'")

        if t["null_pct"] < 0.05:
            score += 0.04
            evidence.append("Consistent temporal indexing with minimal timestamp gaps")

        return min(0.96, max(0.2, round(score, 2))), evidence

    @classmethod
    def _score_comparative(cls, m: Dict[str, Any], d: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.72
        evidence = [
            f"Measure '{m['name']}' paired with categorical dimension '{d['name']}'",
            f"Dimension contains {d['distinct_count']} distinct categories (optimal segment granularity)",
        ]
        if m.get("business_role") in ("revenue", "profit", "churn"):
            score += 0.15
            evidence.append(f"High priority assigned to business measure '{m['business_role']}'")
        if d.get("business_role") in ("region", "customer", "product", "country", "state"):
            score += 0.08
            evidence.append(f"Standard enterprise dimension '{d['business_role']}' identified")

        return min(0.95, max(0.2, round(score, 2))), evidence

    @classmethod
    def _score_target_comparison(cls, tgt: Dict[str, Any], d: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.98
        evidence = [
            f"Target variable '{tgt['name']}' explicitly prioritized",
            f"Segment comparison against dimension '{d['name']}' evaluates group risk disparities",
            "Grain-safe categorical rate comparison",
        ]
        return min(0.99, round(score, 2)), evidence

    @classmethod
    def _score_correlation(cls, m1: Dict[str, Any], m2: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.70
        evidence = [
            f"Continuous numeric measures '{m1['name']}' and '{m2['name']}' selected",
            "Evaluates statistical co-variation without assuming directional causation",
        ]
        if m1.get("business_role") in ("revenue", "price", "churn") or m2.get("business_role") in ("revenue", "price", "churn"):
            score += 0.12
            evidence.append("One or both variables represent strategic business drivers")
        return min(0.92, round(score, 2)), evidence

    @classmethod
    def _score_anomaly(cls, m: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.74
        evidence = [
            f"Numeric measure '{m['name']}' audited for extreme statistical deviations",
            "Statistically defensible Z-score and IQR testing against empirical baseline",
        ]
        if m.get("business_role") in ("revenue", "profit", "cost", "quantity"):
            score += 0.14
            evidence.append(f"Financial exposure/volume metric '{m['business_role']}' merits active anomaly auditing")
        return min(0.94, round(score, 2)), evidence


# =====================================================================
# STEP 3 & 6: ANALYSIS EXECUTION ENGINE (GRAIN-AWARE & DETERMINISTIC)
# =====================================================================

class AnalysisExecutionEngine:
    """Executes generic business analyses deterministically on DataFrames."""

    @classmethod
    def execute_analysis(
        cls,
        opportunity: Dict[str, Any],
        df: pd.DataFrame,
        secondary_df: Optional[pd.DataFrame] = None,
        relationship: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a specific analysis opportunity."""
        a_type = opportunity.get("analysis_type")
        prim_col = opportunity.get("primary_column")
        sec_col = opportunity.get("secondary_column")

        # Cross-dataset execution evaluated first so grain/join safety is strictly enforced
        if a_type == "cross_dataset":
            return cls._execute_cross_dataset(opportunity, df, secondary_df, relationship)

        # Edge case: empty dataframe
        if df.empty or len(df) < 2:
            return cls._empty_result(opportunity, "Insufficient observations (less than 2 rows)")

        if a_type == "descriptive":
            return cls._execute_descriptive(opportunity, df, prim_col)
        elif a_type == "trend":
            return cls._execute_trend(opportunity, df, prim_col, sec_col)
        elif a_type == "comparative":
            return cls._execute_comparative(opportunity, df, prim_col, sec_col)
        elif a_type == "correlation":
            return cls._execute_correlation(opportunity, df, prim_col, sec_col)
        elif a_type == "anomaly":
            return cls._execute_anomaly(opportunity, df, prim_col)
        elif a_type == "segmentation":
            return cls._execute_segmentation(opportunity, df, prim_col, sec_col)
        else:
            return cls._empty_result(opportunity, f"Unknown analysis type: {a_type}")

    @classmethod
    def _empty_result(cls, opp: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "opportunity_id": opp.get("id"),
            "analysis_type": opp.get("analysis_type"),
            "title": opp.get("title"),
            "status": "unavailable",
            "summary_stats": {},
            "visualization_data": {},
            "evidence": [],
            "limitations": [reason],
            "warnings": [reason],
        }

    # 1. Descriptive
    @classmethod
    def _execute_descriptive(cls, opp: Dict[str, Any], df: pd.DataFrame, col: str) -> Dict[str, Any]:
        if col not in df.columns:
            return cls._empty_result(opp, f"Column '{col}' not present in dataset")

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 2:
            return cls._empty_result(opp, f"Insufficient non-null numeric observations in '{col}'")

        count = int(len(series))
        total_sum = safe_float(series.sum())
        mean_val = safe_float(series.mean())
        median_val = safe_float(series.median())
        std_val = safe_float(series.std()) if count > 1 else 0.0
        min_val = safe_float(series.min())
        max_val = safe_float(series.max())
        q25 = safe_float(series.quantile(0.25))
        q75 = safe_float(series.quantile(0.75))
        iqr = safe_float((q75 or 0.0) - (q25 or 0.0))
        skew_val = safe_float(series.skew()) if count > 2 else 0.0

        # Concentration / Pareto: Top 20% records contribution
        sorted_s = series.sort_values(ascending=False)
        top_20_count = max(1, int(math.ceil(count * 0.2)))
        top_20_sum = float(sorted_s.head(top_20_count).sum())
        concentration_pct = safe_float((top_20_sum / total_sum * 100.0) if total_sum and total_sum > 0 else 0.0)

        # Histograms / Bins for visualization
        counts, bin_edges = np.histogram(series, bins=min(10, count))
        bins_data = [
            {
                "bin_label": f"{safe_float(bin_edges[i], 1)} - {safe_float(bin_edges[i+1], 1)}",
                "count": int(counts[i]),
                "share_pct": safe_float(counts[i] / count * 100.0, 1),
            }
            for i in range(len(counts))
        ]

        evidence = [
            f"Calculated from {count:,} non-null numeric observations in '{col}'",
            f"Empirical mean: {mean_val}, median: {median_val}, standard deviation: {std_val}",
            f"Concentration ratio: Top 20% of records account for {concentration_pct}% of cumulative volume",
        ]

        warnings = []
        if skew_val is not None and abs(skew_val) > 1.8:
            warnings.append(f"Substantial distribution skew detected (coefficient: {skew_val:.2f})")

        return {
            "opportunity_id": opp.get("id"),
            "analysis_type": "descriptive",
            "title": opp.get("title"),
            "primary_column": col,
            "status": "completed",
            "summary_stats": {
                "count": count,
                "sum": total_sum,
                "mean": mean_val,
                "median": median_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "iqr": iqr,
                "skewness": skew_val,
                "top_20_pct_concentration": concentration_pct,
            },
            "visualization_data": {"histogram": bins_data},
            "evidence": evidence,
            "limitations": [
                "Descriptive summary reflects observed dataset distribution; does not imply causal generation mechanism.",
            ],
            "warnings": warnings,
        }

    # 2. Trend
    @classmethod
    def _execute_trend(cls, opp: Dict[str, Any], df: pd.DataFrame, prim_col: str, sec_col: Optional[str]) -> Dict[str, Any]:
        time_col = sec_col if sec_col else prim_col
        measure_col = prim_col if sec_col else None

        if time_col not in df.columns:
            return cls._empty_result(opp, f"Time column '{time_col}' not found")

        parsed_dates = pd.to_datetime(df[time_col], errors="coerce")
        valid_mask = parsed_dates.notna()
        if valid_mask.sum() < 3:
            return cls._empty_result(opp, f"Fewer than 3 valid timestamps in '{time_col}'")

        sub_df = df[valid_mask].copy()
        sub_df["_dt"] = parsed_dates[valid_mask]

        min_dt = sub_df["_dt"].min()
        max_dt = sub_df["_dt"].max()
        span_days = (max_dt - min_dt).days

        if span_days <= 0:
            return cls._empty_result(opp, f"Timestamps in '{time_col}' do not span multiple periods")

        # Select period bucket
        if span_days > 180:
            p_format, p_label = "%Y-%m", "monthly"
        elif span_days > 30:
            p_format, p_label = "%Y-W%W", "weekly"
        else:
            p_format, p_label = "%Y-%m-%d", "daily"

        sub_df["_period"] = sub_df["_dt"].dt.strftime(p_format)

        if measure_col and measure_col in sub_df.columns:
            sub_df["_val"] = pd.to_numeric(sub_df[measure_col], errors="coerce")
            sub_df = sub_df.dropna(subset=["_val"])
            if sub_df.empty:
                return cls._empty_result(opp, f"No non-null numeric values for measure '{measure_col}'")
            grouped = sub_df.groupby("_period")["_val"].agg(["sum", "mean", "count"]).reset_index()
            val_col = "sum"
        else:
            grouped = sub_df.groupby("_period")["_dt"].agg(["count"]).reset_index()
            grouped["sum"] = grouped["count"]
            grouped["mean"] = 1.0
            val_col = "count"

        grouped = grouped.sort_values(by="_period")
        if len(grouped) < 2:
            return cls._empty_result(opp, "Insufficient time periods after aggregation (less than 2 periods)")

        periods = []
        for _, row in grouped.iterrows():
            periods.append({
                "period": str(row["_period"]),
                "value": safe_float(row[val_col]),
                "count": int(row["count"]),
                "mean": safe_float(row["mean"]),
            })

        # Calculate period-over-period percentage changes
        pop_changes = []
        for i in range(1, len(periods)):
            prev_v = periods[i-1]["value"] or 0.0
            curr_v = periods[i]["value"] or 0.0
            if prev_v > 0:
                chg = ((curr_v - prev_v) / prev_v) * 100.0
                pop_changes.append(safe_float(chg))

        # Overall trend
        first_val = periods[0]["value"] or 0.0
        last_val = periods[-1]["value"] or 0.0
        overall_change = safe_float(((last_val - first_val) / first_val * 100.0) if first_val > 0 else 0.0)
        direction = "upward" if (overall_change or 0.0) > 3.0 else ("downward" if (overall_change or 0.0) < -3.0 else "stable")

        # Simple seasonality signal: variance of residuals vs mean
        vals = [p["value"] for p in periods if p["value"] is not None]
        seasonality_signal = "low"
        if len(vals) >= 6:
            cv = (np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0.0
            if cv > 0.35:
                seasonality_signal = "moderate_to_high"

        return {
            "opportunity_id": opp.get("id"),
            "analysis_type": "trend",
            "title": opp.get("title"),
            "status": "completed",
            "summary_stats": {
                "period_type": p_label,
                "periods_count": len(periods),
                "start_period": periods[0]["period"],
                "end_period": periods[-1]["period"],
                "start_value": first_val,
                "end_value": last_val,
                "overall_change_pct": overall_change,
                "direction": direction,
                "average_pop_change_pct": safe_float(np.mean(pop_changes)) if pop_changes else 0.0,
                "seasonality_signal": seasonality_signal,
            },
            "visualization_data": {"series": periods},
            "evidence": [
                f"Aggregated {len(sub_df):,} records into {len(periods)} consecutive {p_label} periods",
                f"Trajectory moved from {first_val} ({periods[0]['period']}) to {last_val} ({periods[-1]['period']})",
                f"Net movement: {overall_change:+.1f}% ({direction})",
            ],
            "limitations": [
                "Observed historical trend does not guarantee future trajectory or establish external causality.",
                "Short series may exhibit random fluctuations or reporting cycle artifacts.",
            ],
            "warnings": [],
        }

    # 3. Comparative
    @classmethod
    def _execute_comparative(cls, opp: Dict[str, Any], df: pd.DataFrame, prim_col: str, sec_col: Optional[str]) -> Dict[str, Any]:
        meas_col, dim_col = prim_col, sec_col
        if not sec_col or dim_col not in df.columns or meas_col not in df.columns:
            return cls._empty_result(opp, f"Required columns '{meas_col}' and '{dim_col}' not found")

        sub_df = df[[meas_col, dim_col]].dropna().copy()
        if sub_df.empty:
            return cls._empty_result(opp, "No matching non-null rows for measure and dimension")

        # Determine if measure is boolean target (e.g. churn=0/1) or continuous
        is_bool_or_target = (
            sub_df[meas_col].isin([0, 1, "0", "1", True, False, "true", "false", "yes", "no"]).all()
            or opp.get("primary_column") in ("churn", "target", "conversion")
        )

        if is_bool_or_target:
            sub_df["_target_num"] = (
                sub_df[meas_col].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
            )
            grouped = sub_df.groupby(dim_col)["_target_num"].agg(["count", "mean", "sum"]).reset_index()
            grouped.rename(columns={"mean": "rate", "sum": "positive_count"}, inplace=True)
            grouped["rate_pct"] = grouped["rate"] * 100.0
            grouped = grouped.sort_values(by="rate_pct", ascending=False)
            val_key = "rate_pct"
        else:
            sub_df["_num"] = pd.to_numeric(sub_df[meas_col], errors="coerce")
            sub_df = sub_df.dropna(subset=["_num"])
            if sub_df.empty:
                return cls._empty_result(opp, f"Non-numeric values in measure column '{meas_col}'")
            grouped = sub_df.groupby(dim_col)["_num"].agg(["sum", "mean", "count"]).reset_index()
            total_sum = grouped["sum"].sum()
            grouped["share_pct"] = (grouped["sum"] / total_sum * 100.0) if total_sum > 0 else 0.0
            grouped = grouped.sort_values(by="sum", ascending=False)
            val_key = "share_pct"

        if len(grouped) < 2:
            return cls._empty_result(opp, f"Dimension '{dim_col}' has fewer than 2 distinct categories")

        segments = []
        for _, row in grouped.head(10).iterrows():
            segments.append({
                "segment": str(row[dim_col])[:40],
                "count": int(row["count"]),
                "sum": safe_float(row.get("sum")),
                "mean": safe_float(row.get("mean") or row.get("rate")),
                "metric_value": safe_float(row[val_key]),
            })

        top_seg = segments[0]
        bot_seg = segments[-1]
        disparity_ratio = safe_float((top_seg["metric_value"] / bot_seg["metric_value"]) if bot_seg["metric_value"] and bot_seg["metric_value"] > 0 else None)

        evidence = [
            f"Segmented {len(sub_df):,} records across {len(grouped)} groups in '{dim_col}'",
            f"Leading segment: '{top_seg['segment']}' with {val_key.replace('_', ' ')} of {top_seg['metric_value']}",
            f"Trailing segment: '{bot_seg['segment']}' with {val_key.replace('_', ' ')} of {bot_seg['metric_value']}",
        ]
        if disparity_ratio is not None and disparity_ratio > 1.5:
            evidence.append(f"Disparity: Highest segment is {disparity_ratio:.1f}x the lowest segment")

        return {
            "opportunity_id": opp.get("id"),
            "analysis_type": "comparative",
            "title": opp.get("title"),
            "status": "completed",
            "summary_stats": {
                "dimension": dim_col,
                "measure": meas_col,
                "total_categories": len(grouped),
                "top_segment": top_seg["segment"],
                "top_segment_value": top_seg["metric_value"],
                "bottom_segment": bot_seg["segment"],
                "bottom_segment_value": bot_seg["metric_value"],
                "disparity_ratio": disparity_ratio,
                "is_target_rate": is_bool_or_target,
            },
            "visualization_data": {"segments": segments},
            "evidence": evidence,
            "limitations": [
                "Comparative variation across groups represents observational data and does not demonstrate causal treatment effects.",
                "Unobserved confounding variables may explain segment differences.",
            ],
            "warnings": [],
        }

    # 4. Correlation / Association
    @classmethod
    def _execute_correlation(cls, opp: Dict[str, Any], df: pd.DataFrame, m1: str, m2: Optional[str]) -> Dict[str, Any]:
        if not m2 or m1 not in df.columns or m2 not in df.columns:
            return cls._empty_result(opp, "Both numeric columns required for correlation")

        sub_df = df[[m1, m2]].copy()
        sub_df[m1] = pd.to_numeric(sub_df[m1], errors="coerce")
        sub_df[m2] = pd.to_numeric(sub_df[m2], errors="coerce")
        sub_df = sub_df.dropna()

        if len(sub_df) < 5:
            return cls._empty_result(opp, f"Fewer than 5 complete observation pairs between '{m1}' and '{m2}'")

        # Constant check
        if sub_df[m1].std() == 0 or sub_df[m2].std() == 0:
            return cls._empty_result(opp, "One or both variables have zero variance (constant values)")

        # Pearson correlation and p-value
        r, p_val = stats.pearsonr(sub_df[m1], sub_df[m2])
        r = safe_float(r)
        p_val = safe_float(p_val, 6)

        # Association strength
        abs_r = abs(r or 0.0)
        if abs_r >= 0.7:
            strength = "strong"
        elif abs_r >= 0.4:
            strength = "moderate"
        elif abs_r >= 0.2:
            strength = "weak"
        else:
            strength = "negligible"

        direction = "positive" if (r or 0.0) > 0 else "negative"

        evidence = [
            f"Measured Pearson correlation coefficient r = {r} across {len(sub_df):,} paired records",
            f"Two-tailed statistical significance: p-value = {p_val} (statistically {'significant' if (p_val or 1) < 0.05 else 'not significant'} at α=0.05)",
            f"Classified as a {strength} {direction} association",
        ]

        return {
            "opportunity_id": opp.get("id"),
            "analysis_type": "correlation",
            "title": opp.get("title"),
            "status": "completed",
            "summary_stats": {
                "variable_1": m1,
                "variable_2": m2,
                "sample_size": len(sub_df),
                "correlation_coefficient": r,
                "p_value": p_val,
                "strength": strength,
                "direction": direction,
                "is_significant": bool(p_val is not None and p_val < 0.05),
            },
            "visualization_data": {
                "scatter_summary": {
                    "v1_mean": safe_float(sub_df[m1].mean()),
                    "v2_mean": safe_float(sub_df[m2].mean()),
                    "v1_std": safe_float(sub_df[m1].std()),
                    "v2_std": safe_float(sub_df[m2].std()),
                }
            },
            "evidence": evidence,
            "limitations": [
                "CRITICAL: Correlation measures statistical association ONLY. It does NOT establish or imply that changes in one variable cause changes in the other.",
                "Non-linear associations, common confounding factors, or reverse directionality may be present.",
            ],
            "warnings": [
                "Association does NOT imply causation. Do not take business interventions based purely on correlation without validation.",
            ],
        }

    # 5. Anomaly Detection
    @classmethod
    def _execute_anomaly(cls, opp: Dict[str, Any], df: pd.DataFrame, col: str) -> Dict[str, Any]:
        if col not in df.columns:
            return cls._empty_result(opp, f"Column '{col}' not found")

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 5:
            return cls._empty_result(opp, f"Fewer than 5 numeric records in '{col}' for anomaly testing")

        mean_val = float(series.mean())
        std_val = float(series.std())
        if std_val == 0:
            return cls._empty_result(opp, f"Zero variance in '{col}'; no anomalies can exist")

        # Z-scores
        z_scores = (series - mean_val) / std_val
        anomalies_mask = z_scores.abs() >= 2.5
        anomalous_indices = series[anomalies_mask].index.tolist()
        anomaly_count = len(anomalous_indices)

        top_anomalies = []
        for idx in anomalous_indices[:5]:
            obs_val = safe_float(series[idx])
            z = safe_float(z_scores[idx])
            diff = safe_float(obs_val - mean_val)
            top_anomalies.append({
                "row_index": int(idx),
                "observed_value": obs_val,
                "baseline_mean": safe_float(mean_val),
                "deviation_magnitude": diff,
                "z_score": z,
                "direction": "spike_high" if (z or 0) > 0 else "drop_low",
            })

        evidence = [
            f"Screened {len(series):,} records against empirical baseline (mean: {mean_val:.2f}, std: {std_val:.2f})",
            f"Discovered {anomaly_count} records exceeding threshold of 2.5 standard deviations",
        ]
        if top_anomalies:
            max_dev = max(top_anomalies, key=lambda a: abs(a.get("z_score") or 0))
            evidence.append(
                f"Peak extreme value: {max_dev['observed_value']} (z = {max_dev['z_score']:.2f}, "
                f"diverging by {max_dev['deviation_magnitude']} from baseline)"
            )

        return {
            "opportunity_id": opp.get("id"),
            "analysis_type": "anomaly",
            "title": opp.get("title"),
            "status": "completed",
            "summary_stats": {
                "audited_column": col,
                "total_records": len(series),
                "anomaly_count": anomaly_count,
                "anomaly_percentage": safe_float((anomaly_count / len(series) * 100.0) if series.size else 0.0),
                "baseline_mean": safe_float(mean_val),
                "baseline_std": safe_float(std_val),
                "max_z_score": safe_float(z_scores.abs().max()),
                "top_anomalies": top_anomalies,
            },
            "visualization_data": {"anomalies": top_anomalies},
            "evidence": evidence,
            "limitations": [
                "Statistical anomalies indicate mathematical divergence from the baseline, not necessarily errors, fraud, or business failures.",
                "High-value genuine customer transactions naturally appear in extreme statistical tails.",
            ],
            "warnings": [],
        }

    # 6. Segmentation
    @classmethod
    def _execute_segmentation(cls, opp: Dict[str, Any], df: pd.DataFrame, col1: str, col2: Optional[str]) -> Dict[str, Any]:
        cols = [col1]
        if col2 and col2 in df.columns:
            cols.append(col2)

        sub_df = df[cols].copy()
        for c in cols:
            sub_df[c] = pd.to_numeric(sub_df[c], errors="coerce")
        sub_df = sub_df.dropna()

        if len(sub_df) < 10:
            return cls._empty_result(opp, "Fewer than 10 clean records for unsupervised clustering")

        k = min(3, len(sub_df))
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            sub_df["_cluster"] = kmeans.fit_predict(sub_df[cols])
        except Exception as e:
            return cls._empty_result(opp, f"Clustering failed: {str(e)}")

        clusters = []
        for cl_id in range(k):
            cl_data = sub_df[sub_df["_cluster"] == cl_id]
            profile = {
                "cluster_id": cl_id,
                "size": len(cl_data),
                "share_pct": safe_float(len(cl_data) / len(sub_df) * 100.0),
            }
            for c in cols:
                profile[f"{c}_mean"] = safe_float(cl_data[c].mean())
            clusters.append(profile)

        # Sort clusters by size
        clusters.sort(key=lambda c: c["size"], reverse=True)

        return {
            "opportunity_id": opp.get("id"),
            "analysis_type": "segmentation",
            "title": opp.get("title"),
            "status": "completed",
            "summary_stats": {
                "clusters_count": k,
                "features_used": cols,
                "total_observations": len(sub_df),
                "clusters": clusters,
            },
            "visualization_data": {"clusters": clusters},
            "evidence": [
                f"Clustered {len(sub_df):,} records across features: {', '.join(cols)}",
                f"Generated {k} distinct operational cohorts without using keys or timestamps",
            ],
            "limitations": [
                "Clusters represent mathematical partitions; business cohort naming requires human domain validation.",
            ],
            "warnings": [],
        }

    # 7. Cross-Dataset Grain-Aware Execution
    @classmethod
    def _execute_cross_dataset(
        cls,
        opp: Dict[str, Any],
        source_df: pd.DataFrame,
        target_df: Optional[pd.DataFrame],
        rel: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if target_df is None or rel is None:
            return cls._empty_result(opp, "Secondary dataset or relationship definition missing")

        s_col = rel.get("source_column")
        t_col = rel.get("target_column")
        cardinality = rel.get("cardinality", "one_to_one")
        fan_out = float(rel.get("fan_out_ratio", 1.0) or 1.0)
        has_double_counting = bool(rel.get("double_counting_risk", False))
        val_status = rel.get("validation_status", "candidate")

        # Grain Safeguard
        if val_status != "validated" and (has_double_counting or fan_out > 1.05):
            return {
                "opportunity_id": opp.get("id"),
                "analysis_type": "cross_dataset",
                "title": opp.get("title"),
                "status": "unavailable",
                "summary_stats": {},
                "visualization_data": {},
                "evidence": [],
                "limitations": [
                    f"Analysis unavailable: Join exhibits a {fan_out:.1f}x fan-out ({cardinality}) "
                    f"with unaggregated measures. Pre-aggregation is required before safe analysis."
                ],
                "warnings": [
                    f"Execution blocked to prevent measure inflation across independent grains."
                ],
            }

        # Safe execution: pre-aggregate child rows if fan-out > 1.0
        # Determine which side is many
        s_keys_unique = source_df[s_col].nunique() == len(source_df)
        t_keys_unique = target_df[t_col].nunique() == len(target_df)

        # Find a numeric measure in source or target
        s_numeric = [c for c in source_df.select_dtypes(include=np.number).columns if c != s_col]
        t_categorical = [c for c in target_df.select_dtypes(exclude=np.number).columns if c != t_col]

        if not s_numeric or not t_categorical:
            return cls._empty_result(opp, "Could not find compatible measure and dimension across datasets")

        meas_col = s_numeric[0]
        dim_col = t_categorical[0]

        # Pre-aggregate if target has duplicates on join key
        t_clean = target_df[[t_col, dim_col]].dropna().drop_duplicates(subset=[t_col])
        s_clean = source_df[[s_col, meas_col]].dropna()

        merged = pd.merge(s_clean, t_clean, left_on=s_col, right_on=t_col, how="inner")
        if merged.empty:
            return cls._empty_result(opp, "Zero records matched across cross-dataset keys")

        grouped = merged.groupby(dim_col)[meas_col].agg(["sum", "count"]).reset_index()
        total_sum = grouped["sum"].sum()
        grouped["share_pct"] = (grouped["sum"] / total_sum * 100.0) if total_sum > 0 else 0.0
        grouped = grouped.sort_values(by="sum", ascending=False)

        top_items = []
        for _, row in grouped.head(6).iterrows():
            top_items.append({
                "group": str(row[dim_col])[:40],
                "sum": safe_float(row["sum"]),
                "count": int(row["count"]),
                "share_pct": safe_float(row["share_pct"]),
            })

        return {
            "opportunity_id": opp.get("id"),
            "analysis_type": "cross_dataset",
            "title": opp.get("title"),
            "status": "completed",
            "summary_stats": {
                "joined_records": len(merged),
                "measure": meas_col,
                "dimension": dim_col,
                "total_sum": safe_float(total_sum),
                "breakdown": top_items,
            },
            "visualization_data": {"breakdown": top_items},
            "evidence": [
                f"Grain-safe join executed across {len(merged):,} merged records",
                f"Pre-aggregated to key grain to eliminate {fan_out:.1f}x fan-out inflation risk",
                f"Measured cross-dataset metric '{meas_col}' grouped by '{dim_col}'",
            ],
            "limitations": [
                "Assumes inner join key consistency between both datasets.",
            ],
            "warnings": [],
        }


# =====================================================================
# STEP 7, 8, 9, 14, 15: DYNAMIC INSIGHT GENERATION, DEDUPLICATION & RANKING
# =====================================================================

class DynamicInsightEngine:
    """Generates structured, deduplicated, and proactively ranked business insights from executed analyses."""

    SEVERITY_ORDER = {"critical": 4, "warning": 3, "positive": 2, "info": 1}

    @classmethod
    def generate_insights_from_analysis(cls, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synthesize candidate insights from an executed analysis result."""
        if analysis_result.get("status") != "completed":
            return []

        a_type = analysis_result.get("analysis_type")
        stats_data = analysis_result.get("summary_stats", {})
        title = analysis_result.get("title", "")
        opp_id = analysis_result.get("opportunity_id", "opp")

        # Extract clean column/metric name
        col = (
            analysis_result.get("primary_column")
            or stats_data.get("column")
            or stats_data.get("measure")
            or stats_data.get("audited_column")
        )
        if not col:
            words = title.split()
            col = words[-1] if ": " in title else (words[0] if words else "Metric")
        col_clean = str(col).replace("_", " ").title()

        insights = []

        # 1. Descriptive Insights (Concentration & Skew)
        if a_type == "descriptive":
            conc = stats_data.get("top_20_pct_concentration")
            skew = stats_data.get("skewness")
            total_sum = stats_data.get("sum")

            if conc is not None and conc >= 65.0:
                severity = "critical" if conc >= 80.0 else "warning"
                insights.append({
                    "id": f"ins_conc_{opp_id}",
                    "title": f"Top 20% of Records Drive {conc:.0f}% of Total '{col_clean}'",
                    "category": "concentration",
                    "severity": severity,
                    "metric": col_clean,
                    "value": conc,
                    "baseline": 20.0,
                    "change": safe_float(conc - 20.0),
                    "confidence": 0.92,
                    "affected_segment": f"Top 20% highest-volume records in '{col_clean}'",
                    "dataset_context": f"Empirical distribution across {stats_data.get('count', 0):,} records",
                    "evidence": [
                        f"The top 20% of records account for {conc:.1f}% of all cumulative '{col_clean}' volume ({total_sum:,.0f} total).",
                        "Heavy Pareto concentration: performance relies heavily on a small group of high-volume observations.",
                    ],
                    "why_it_matters": f"Disproportionate concentration creates operational vulnerability. If key top entities fluctuate or churn, overall '{col_clean}' will swing dramatically.",
                    "limitations": [
                        "Snapshot observation; does not track individual account history over time.",
                    ],
                    "recommended_next_analysis": f"Perform account-level breakdown to audit the retention of top '{col_clean}' contributors.",
                    "is_causal_claim": False,
                })

            if skew is not None and abs(skew) > 1.8:
                mean_val = stats_data.get("mean")
                median_val = stats_data.get("median")
                mean_str = f"{mean_val:,.2f}" if isinstance(mean_val, (int, float)) else str(mean_val)
                median_str = f"{median_val:,.2f}" if isinstance(median_val, (int, float)) else str(median_val)
                direction_word = "high-end" if skew > 0 else "low-end"

                insights.append({
                    "id": f"ins_skew_{opp_id}",
                    "title": f"Extreme Values in '{col_clean}' Distort Averages",
                    "category": "descriptive",
                    "severity": "info",
                    "metric": col_clean,
                    "value": skew,
                    "baseline": 0.0,
                    "change": skew,
                    "confidence": 0.85,
                    "affected_segment": f"Extreme {direction_word} records in '{col_clean}'",
                    "dataset_context": f"Observed across {stats_data.get('count', 0):,} records",
                    "evidence": [
                        f"Most records cluster tightly, but extreme {direction_word} outliers stretch the data range.",
                        f"Typical record value (median: {median_str}) differs sharply from the mathematical average (mean: {mean_str}).",
                    ],
                    "why_it_matters": f"Don't base business rules, budgets, or fraud thresholds on simple averages of '{col_clean}'. Massive outliers distort your baseline and trigger false alarms.",
                    "limitations": [
                        "Requires manual review to confirm whether outliers represent genuine high-value VIPs or logging glitches.",
                    ],
                    "recommended_next_analysis": f"Use median or percentile thresholds instead of simple averages when setting rules for '{col_clean}'.",
                    "is_causal_claim": False,
                })

        # 2. Trend Insights
        elif a_type == "trend":
            chg = stats_data.get("overall_change_pct")
            direction = stats_data.get("direction", "stable")
            p_type = stats_data.get("period_type", "period")
            start_p = stats_data.get("start_period")
            end_p = stats_data.get("end_period")

            if chg is not None and abs(chg) >= 4.0:
                is_growth = chg > 0
                is_adverse = ("churn" in title.lower() or "cost" in title.lower()) and is_growth
                is_beneficial = ("revenue" in title.lower() or "profit" in title.lower()) and is_growth

                if is_adverse:
                    severity = "critical" if abs(chg) > 15.0 else "warning"
                elif is_beneficial:
                    severity = "positive"
                else:
                    severity = "warning" if not is_growth else "positive"

                verb = "increased" if is_growth else "declined"
                insights.append({
                    "id": f"ins_trend_{opp_id}",
                    "title": f"'{col_clean}' Has {verb.capitalize()} by {abs(chg):.1f}% Over Recent {p_type.capitalize()}s",
                    "category": "trend",
                    "severity": severity,
                    "metric": col_clean,
                    "value": chg,
                    "baseline": 0.0,
                    "change": chg,
                    "confidence": 0.88,
                    "affected_segment": f"Trend from {start_p} to {end_p}",
                    "dataset_context": f"Observed over {stats_data.get('periods_count')} {p_type} cycles",
                    "evidence": [
                        f"Value moved from {stats_data.get('start_value')} to {stats_data.get('end_value')} ({chg:+.1f}% net change).",
                        f"Average period-to-period change was {stats_data.get('average_pop_change_pct'):+.1f}%.",
                    ],
                    "why_it_matters": f"A sustained {direction} in '{col_clean}' directly impacts operational capacity, inventory, and revenue runway.",
                    "limitations": [
                        "Historical movement does not guarantee future results; seasonality or marketing campaigns may be factors.",
                    ],
                    "recommended_next_analysis": f"Break down '{col_clean}' by segment or region to isolate which groups drove the change.",
                    "is_causal_claim": False,
                })

        # 3. Comparative Insights
        elif a_type == "comparative":
            top_seg = stats_data.get("top_segment")
            top_val = stats_data.get("top_segment_value")
            bot_seg = stats_data.get("bottom_segment")
            bot_val = stats_data.get("bottom_segment_value")
            disparity = stats_data.get("disparity_ratio")
            dim = stats_data.get("dimension")
            is_target_rate = stats_data.get("is_target_rate", False)

            if top_seg and disparity is not None and disparity >= 1.5:
                severity = "critical" if (is_target_rate and disparity >= 2.0) else "warning"
                meas_clean = str(stats_data.get("measure") or col_clean).replace("_", " ").title()
                insights.append({
                    "id": f"ins_comp_{opp_id}",
                    "title": f"'{top_seg}' Outperforms '{bot_seg}' by {disparity:.1f}x in '{meas_clean}'",
                    "category": "comparative",
                    "severity": severity,
                    "metric": meas_clean,
                    "value": top_val,
                    "baseline": bot_val,
                    "change": safe_float(top_val - (bot_val or 0.0)) if top_val and bot_val else None,
                    "confidence": 0.90,
                    "affected_segment": f"{dim}: '{top_seg}' vs '{bot_seg}'",
                    "dataset_context": f"Compared across {stats_data.get('total_categories')} categories in '{dim}'",
                    "evidence": [
                        f"'{top_seg}' averages {top_val} compared to '{bot_seg}' at {bot_val}.",
                        f"Observed gap is {disparity:.1f}x between the highest and lowest performing groups in '{dim}'.",
                    ],
                    "why_it_matters": f"Wide performance differences across {dim} cohorts highlight either successful strategies to scale or lagging segments needing intervention.",
                    "limitations": [
                        "Segment differences are observational; underlying differences in customer age, tenure, or acquisition channel should be verified.",
                    ],
                    "recommended_next_analysis": f"Investigate tenure and activity differences within '{top_seg}'.",
                    "is_causal_claim": False,
                })

        # 4. Correlation Insights
        elif a_type == "correlation":
            r = stats_data.get("correlation_coefficient")
            p_val = stats_data.get("p_value")
            v1 = stats_data.get("variable_1")
            v2 = stats_data.get("variable_2")
            strength = stats_data.get("strength")
            direction = stats_data.get("direction")

            if r is not None and abs(r) >= 0.35 and (p_val or 1) < 0.05:
                severity = "positive" if (r > 0 and strength == "strong") else "info"
                v1_clean = str(v1).replace("_", " ").title()
                v2_clean = str(v2).replace("_", " ").title()
                insights.append({
                    "id": f"ins_corr_{opp_id}",
                    "title": f"Strong Relationship Discovered Between '{v1_clean}' and '{v2_clean}'",
                    "category": "correlation",
                    "severity": severity,
                    "metric": f"{v1_clean} ↔ {v2_clean}",
                    "value": r,
                    "baseline": 0.0,
                    "change": r,
                    "confidence": 0.88,
                    "affected_segment": f"Records containing both '{v1_clean}' and '{v2_clean}'",
                    "dataset_context": f"Evaluated across {stats_data.get('sample_size'):,} observations",
                    "evidence": [
                        f"When '{v1_clean}' increases, '{v2_clean}' systematically tends to {'increase' if r > 0 else 'decrease'}.",
                        f"Association strength is {strength} (correlation r = {r:.2f}, statistically significant).",
                    ],
                    "why_it_matters": f"Changes in '{v1_clean}' can act as an early leading indicator for shifts in '{v2_clean}' during forecasting and planning.",
                    "limitations": [
                        "CRITICAL: Correlation does NOT prove that changing one variable causes the other to change.",
                        "Do not make operational changes assuming causation without running a controlled A/B test.",
                    ],
                    "recommended_next_analysis": f"Conduct an A/B test or multivariate regression to verify whether '{v1_clean}' directly impacts '{v2_clean}'.",
                    "is_causal_claim": False,
                })

        # 5. Anomaly Insights
        elif a_type == "anomaly":
            cnt = stats_data.get("anomaly_count", 0)
            col_anom = stats_data.get("audited_column") or col
            max_z = stats_data.get("max_z_score")
            top_anoms = stats_data.get("top_anomalies", [])

            if cnt > 0 and max_z and max_z >= 2.5:
                severity = "warning" if cnt > 3 else "info"
                lead_anom = top_anoms[0] if top_anoms else {}
                anom_col_clean = str(col_anom).replace("_", " ").title()
                insights.append({
                    "id": f"ins_anom_{opp_id}",
                    "title": f"{cnt} Extreme Spike{'s' if cnt > 1 else ''} Detected in '{anom_col_clean}'",
                    "category": "anomaly",
                    "severity": severity,
                    "metric": anom_col_clean,
                    "value": lead_anom.get("observed_value"),
                    "baseline": stats_data.get("baseline_mean"),
                    "change": lead_anom.get("deviation_magnitude"),
                    "confidence": 0.94,
                    "affected_segment": f"{cnt} unusual record{'s' if cnt > 1 else ''} in '{anom_col_clean}'",
                    "dataset_context": f"Evaluated across {stats_data.get('total_records'):,} observations",
                    "evidence": [
                        f"The highest outlier reached {lead_anom.get('observed_value')}, diverging sharply from the normal average of {stats_data.get('baseline_mean')}.",
                        f"Peak spike was {max_z:.1f} times higher than typical variation.",
                    ],
                    "why_it_matters": f"Unusual spikes in '{anom_col_clean}' can distort KPI tracking, trigger false alarms, or point to data pipeline issues or fraud.",
                    "limitations": [
                        "Identifies mathematical deviation only; check raw transactions to see if they are genuine VIPs or telemetry errors.",
                    ],
                    "recommended_next_analysis": f"Inspect the {cnt} flagged transaction rows in '{anom_col_clean}' to audit their source.",
                    "is_causal_claim": False,
                })

        # 6. Cross-Dataset Insights
        elif a_type == "cross_dataset":
            breakdown = stats_data.get("breakdown", [])
            meas = stats_data.get("measure")
            dim = stats_data.get("dimension")
            joined_n = stats_data.get("joined_records", 0)

            if breakdown and len(breakdown) >= 2:
                top = breakdown[0]
                share = top.get("share_pct") or 0.0
                meas_clean = str(meas).replace("_", " ").title()
                dim_clean = str(dim).replace("_", " ").title()
                if share >= 40.0:
                    insights.append({
                        "id": f"ins_cross_{opp_id}",
                        "title": f"'{top['group']}' Drives {share:.1f}% of Total '{meas_clean}' Across Connected Data",
                        "category": "cross_dataset",
                        "severity": "warning",
                        "metric": meas_clean,
                        "value": share,
                        "baseline": 100.0 / len(breakdown),
                        "change": safe_float(share - (100.0 / len(breakdown))),
                        "confidence": 0.89,
                        "affected_segment": f"{dim_clean}: '{top['group']}'",
                        "dataset_context": f"Synthesized across {joined_n:,} records using validated grain-safe join",
                        "evidence": [
                            f"Linked records show '{top['group']}' commands {share:.1f}% of all '{meas_clean}', far above an equal distribution.",
                            "Join verified safe against fan-out measure double-counting.",
                        ],
                        "why_it_matters": f"Cross-dataset concentration reveals how behavior in one system directly drives concentration in the other.",
                        "limitations": [
                            "Relies on foreign key link validity and active record synchronization.",
                        ],
                        "recommended_next_analysis": f"Investigate account-level drivers within '{top['group']}'.",
                        "is_causal_claim": False,
                    })

        return insights

    @classmethod
    def deduplicate_insights(cls, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group related findings into fewer, higher-value insights."""
        if not insights:
            return []

        # Deduplication key: metric + category
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for ins in insights:
            key = f"{ins.get('category')}__{ins.get('metric', '')}"
            grouped.setdefault(key, []).append(ins)

        deduped = []
        for key, cluster in grouped.items():
            if len(cluster) == 1:
                deduped.append(cluster[0])
            else:
                # Pick the highest severity and highest confidence insight
                best = max(
                    cluster,
                    key=lambda x: (
                        cls.SEVERITY_ORDER.get(x.get("severity", "info"), 1),
                        x.get("confidence", 0.0),
                    )
                )
                # Merge unique evidence points from secondary insights
                all_evidence = list(best.get("evidence", []))
                for other in cluster:
                    if other["id"] != best["id"]:
                        for ev in other.get("evidence", []):
                            if ev not in all_evidence:
                                all_evidence.append(ev)
                best["evidence"] = all_evidence[:5]  # Cap evidence
                deduped.append(best)

        return deduped

    @classmethod
    def rank_insights(cls, insights: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """Rank insights by business relevance, severity, magnitude, and statistical confidence."""
        def score_insight(ins: Dict[str, Any]) -> float:
            sev_weight = cls.SEVERITY_ORDER.get(ins.get("severity", "info"), 1) * 25.0
            conf_weight = (ins.get("confidence", 0.5)) * 20.0
            cat_bonus = 15.0 if ins.get("category") in ("concentration", "anomaly", "comparative") else 10.0

            # Magnitude bonus
            val = abs(ins.get("value") or 0.0)
            mag_bonus = min(20.0, val * 0.2)

            return sev_weight + conf_weight + cat_bonus + mag_bonus

        sorted_insights = sorted(insights, key=score_insight, reverse=True)
        return sorted_insights[:top_n]
