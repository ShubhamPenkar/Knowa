"""Pure ML and data-driven analytical engine for Dynamic Product Intelligence.

Discovers usable measures, dimensions, distributions, trends, breakdowns,
insights, and investigation opportunities from unfamiliar datasets using semantic metadata.
Never relies on hardcoded column names, fixed schemas, or industry assumptions.
Never fabricates causal claims, ROI, or financial projections.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def safe_float(val: Any) -> Optional[float]:
    """Convert value to float if valid and finite, else None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 4)
    except (ValueError, TypeError):
        return None


def compute_column_summary(
    series: pd.Series,
    col_name: str,
    dataset_id: str,
    dataset_name: str,
) -> Optional[Dict[str, Any]]:
    """Compute summary statistics for a numeric measure column."""
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return None

    count = int(len(cleaned))
    total = safe_float(cleaned.sum())
    mean_val = safe_float(cleaned.mean())
    median_val = safe_float(cleaned.median())
    min_val = safe_float(cleaned.min())
    max_val = safe_float(cleaned.max())
    std_val = safe_float(cleaned.std()) if count > 1 else 0.0

    # Skewness
    skew_val = safe_float(cleaned.skew()) if count > 2 else 0.0

    return {
        "id": f"metric_summary_{dataset_id}_{col_name}",
        "label": f"{col_name} Distribution & Summary",
        "category": "summary",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "column_name": col_name,
        "secondary_column": None,
        "aggregation": "distribution",
        "status": "available",
        "reason_unavailable": None,
        "value": mean_val,
        "summary_stats": {
            "count": count,
            "sum": total,
            "mean": mean_val,
            "median": median_val,
            "min": min_val,
            "max": max_val,
            "std": std_val,
            "skewness": skew_val,
        },
        "time_range": None,
        "grouping": None,
        "evidence": [
            f"Computed from {count:,} non-null values in column '{col_name}'",
            f"Range spans from {min_val} to {max_val} (mean: {mean_val}, median: {median_val})",
        ],
        "caveats": (
            [f"{len(series) - count} null or non-numeric values excluded"]
            if len(series) > count else []
        ),
        "warnings": (
            ["Substantial distribution skewness detected (|skew| > 1.5)"]
            if skew_val is not None and abs(skew_val) > 1.5 else []
        ),
    }


def compute_categorical_breakdown(
    df: pd.DataFrame,
    dim_col: str,
    measure_col: Optional[str],
    dataset_id: str,
    dataset_name: str,
    max_groups: int = 10,
) -> Optional[Dict[str, Any]]:
    """Compute breakdown of a measure (or row counts) by a categorical dimension."""
    if dim_col not in df.columns:
        return None

    dim_series = df[dim_col].dropna().astype(str).str.strip()
    if dim_series.empty:
        return None

    distinct_count = dim_series.nunique()
    if distinct_count < 2:
        return None

    if measure_col and measure_col in df.columns:
        num_series = pd.to_numeric(df[measure_col], errors="coerce")
        valid_mask = num_series.notna() & df[dim_col].notna()
        sub_df = df[valid_mask].copy()
        if sub_df.empty:
            return None
        sub_df[measure_col] = pd.to_numeric(sub_df[measure_col], errors="coerce")
        grouped = sub_df.groupby(dim_col)[measure_col].agg(["sum", "count", "mean"]).reset_index()
        grouped = grouped.sort_values(by="sum", ascending=False)
        total_sum = grouped["sum"].sum()

        top_groups = []
        for _, row in grouped.head(max_groups).iterrows():
            g_name = str(row[dim_col])[:40]
            g_sum = safe_float(row["sum"])
            g_count = int(row["count"])
            g_mean = safe_float(row["mean"])
            g_share = safe_float((row["sum"] / total_sum * 100.0) if total_sum and total_sum > 0 else 0.0)
            top_groups.append({
                "group": g_name,
                "sum": g_sum,
                "count": g_count,
                "mean": g_mean,
                "share_pct": g_share,
            })

        metric_id = f"metric_breakdown_{dataset_id}_{measure_col}_by_{dim_col}"
        label = f"{measure_col} Breakdown by {dim_col}"
        agg_name = "sum_by_group"
        evidence_str = f"Aggregated {len(sub_df):,} rows across {distinct_count} distinct categories in '{dim_col}'"
    else:
        # Row count frequency breakdown
        counts = dim_series.value_counts()
        total_count = len(dim_series)
        top_groups = []
        for cat_name, cnt in counts.head(max_groups).items():
            top_groups.append({
                "group": str(cat_name)[:40],
                "sum": cnt,
                "count": cnt,
                "mean": 1.0,
                "share_pct": safe_float(cnt / total_count * 100.0 if total_count else 0.0),
            })
        metric_id = f"metric_frequency_{dataset_id}_{dim_col}"
        label = f"Distribution by {dim_col}"
        agg_name = "count_by_group"
        evidence_str = f"Categorical frequency across {total_count:,} records in '{dim_col}'"

    warnings = []
    if distinct_count > max_groups:
        warnings.append(f"Dimension has {distinct_count} categories; showing top {max_groups} by volume")

    return {
        "id": metric_id,
        "label": label,
        "category": "breakdown",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "column_name": measure_col or dim_col,
        "secondary_column": dim_col if measure_col else None,
        "aggregation": agg_name,
        "status": "available",
        "reason_unavailable": None,
        "value": top_groups[0]["share_pct"] if top_groups else None,
        "summary_stats": {
            "groups_count": distinct_count,
            "top_group": top_groups[0]["group"] if top_groups else None,
            "top_group_share_pct": top_groups[0]["share_pct"] if top_groups else None,
            "breakdown": top_groups,
        },
        "time_range": None,
        "grouping": {
            "dimension": dim_col,
            "groups_count": distinct_count,
            "displayed_groups": len(top_groups),
        },
        "evidence": [evidence_str],
        "caveats": ["Groups with low representation or null categories were pruned"],
        "warnings": warnings,
    }


def compute_temporal_trend(
    df: pd.DataFrame,
    time_col: str,
    measure_col: Optional[str],
    dataset_id: str,
    dataset_name: str,
) -> Optional[Dict[str, Any]]:
    """Compute temporal trend and period-over-period change."""
    if time_col not in df.columns:
        return None

    # Parse timestamps
    parsed_dates = pd.to_datetime(df[time_col], errors="coerce")
    valid_mask = parsed_dates.notna()
    if valid_mask.sum() < 3:
        return None

    sub_df = df[valid_mask].copy()
    sub_df["_dt"] = parsed_dates[valid_mask]

    min_date = sub_df["_dt"].min()
    max_date = sub_df["_dt"].max()
    span_days = (max_date - min_date).days

    if span_days <= 0:
        return None

    # Choose period grouping: 'M' (month) if > 60 days, 'W' (week) if > 14 days, else 'D' (day)
    if span_days > 90:
        period_format = "%Y-%m"
        period_label = "monthly"
    elif span_days > 21:
        period_format = "%Y-W%W"
        period_label = "weekly"
    else:
        period_format = "%Y-%m-%d"
        period_label = "daily"

    sub_df["_period"] = sub_df["_dt"].dt.strftime(period_format)

    if measure_col and measure_col in sub_df.columns:
        sub_df["_num"] = pd.to_numeric(sub_df[measure_col], errors="coerce")
        grouped = sub_df.dropna(subset=["_num"]).groupby("_period")["_num"].agg(["sum", "count", "mean"]).reset_index()
        metric_label = f"{measure_col} {period_label.capitalize()} Trend"
        val_key = "sum"
    else:
        grouped = sub_df.groupby("_period")["_dt"].agg(["count"]).reset_index()
        grouped["sum"] = grouped["count"]
        grouped["mean"] = 1.0
        metric_label = f"Activity Volume {period_label.capitalize()} Trend"
        val_key = "count"

    grouped = grouped.sort_values(by="_period")
    if len(grouped) < 2:
        return None

    periods = []
    for _, row in grouped.iterrows():
        periods.append({
            "period": str(row["_period"]),
            "value": safe_float(row[val_key]),
            "count": int(row["count"]),
        })

    # Growth / Change between first and last period
    first_val = periods[0]["value"] or 0.0
    last_val = periods[-1]["value"] or 0.0
    change_pct = None
    if first_val > 0:
        change_pct = safe_float(((last_val - first_val) / first_val) * 100.0)

    # Direction
    if change_pct is not None:
        direction = "upward" if change_pct > 2.0 else ("downward" if change_pct < -2.0 else "stable")
    else:
        direction = "undetermined"

    return {
        "id": f"metric_trend_{dataset_id}_{measure_col or 'volume'}_over_{time_col}",
        "label": metric_label,
        "category": "trend",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "column_name": measure_col or time_col,
        "secondary_column": time_col,
        "aggregation": f"{val_key}_by_{period_label}",
        "status": "available",
        "reason_unavailable": None,
        "value": change_pct,
        "summary_stats": {
            "period_type": period_label,
            "period_count": len(periods),
            "start_period": periods[0]["period"],
            "end_period": periods[-1]["period"],
            "start_value": first_val,
            "end_value": last_val,
            "change_pct": change_pct,
            "direction": direction,
            "periods": periods[-12:],  # Last 12 periods max
        },
        "time_range": {
            "start": min_date.strftime("%Y-%m-%d"),
            "end": max_date.strftime("%Y-%m-%d"),
            "span_days": span_days,
            "period": period_label,
        },
        "grouping": None,
        "evidence": [
            f"Aggregated {len(sub_df):,} records across {len(periods)} {period_label} periods",
            f"Observed {direction} movement from {periods[0]['period']} to {periods[-1]['period']}",
        ],
        "caveats": ["Temporal trends reflect observed series data and do not establish future projections"],
        "warnings": [],
    }


def compute_top_performers(
    df: pd.DataFrame,
    entity_col: str,
    measure_col: str,
    dataset_id: str,
    dataset_name: str,
    top_n: int = 5,
) -> Optional[Dict[str, Any]]:
    """Compute top and bottom ranking entities by a given measure."""
    if entity_col not in df.columns or measure_col not in df.columns:
        return None

    sub_df = df[[entity_col, measure_col]].dropna().copy()
    sub_df[measure_col] = pd.to_numeric(sub_df[measure_col], errors="coerce")
    sub_df = sub_df.dropna()
    if sub_df.empty:
        return None

    grouped = sub_df.groupby(entity_col)[measure_col].sum().reset_index()
    if len(grouped) < 3:
        return None

    grouped = grouped.sort_values(by=measure_col, ascending=False)
    total_val = grouped[measure_col].sum()

    top_items = []
    for _, row in grouped.head(top_n).iterrows():
        ent_name = str(row[entity_col])[:40]
        val = safe_float(row[measure_col])
        share = safe_float((val / total_val * 100.0) if total_val and total_val > 0 else 0.0)
        top_items.append({"entity": ent_name, "value": val, "share_pct": share})

    bottom_items = []
    for _, row in grouped.tail(top_n).iterrows():
        ent_name = str(row[entity_col])[:40]
        val = safe_float(row[measure_col])
        share = safe_float((val / total_val * 100.0) if total_val and total_val > 0 else 0.0)
        bottom_items.append({"entity": ent_name, "value": val, "share_pct": share})

    top_concentration = sum(item["share_pct"] or 0.0 for item in top_items)

    return {
        "id": f"metric_performers_{dataset_id}_{measure_col}_by_{entity_col}",
        "label": f"Top Performers in {measure_col} by {entity_col}",
        "category": "top_performers",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "column_name": measure_col,
        "secondary_column": entity_col,
        "aggregation": "top_ranking",
        "status": "available",
        "reason_unavailable": None,
        "value": safe_float(top_concentration),
        "summary_stats": {
            "total_entities": len(grouped),
            "top_n_concentration_pct": safe_float(top_concentration),
            "top_performers": top_items,
            "bottom_performers": bottom_items,
        },
        "time_range": None,
        "grouping": {"entity_column": entity_col, "total_entities": len(grouped)},
        "evidence": [
            f"Evaluated {len(grouped):,} unique entities from '{entity_col}'",
            f"Top {len(top_items)} entities account for {top_concentration:.1f}% of total {measure_col}",
        ],
        "caveats": ["Entity rankings represent recorded data totals"],
        "warnings": (
            [f"Heavy concentration: top {len(top_items)} entities control over 50% of total volume"]
            if top_concentration > 50.0 else []
        ),
    }


def compute_cross_dataset_metric(
    source_df: pd.DataFrame,
    source_meta: Dict[str, Any],
    source_grain: Dict[str, Any],
    target_df: pd.DataFrame,
    target_meta: Dict[str, Any],
    target_grain: Dict[str, Any],
    rel: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute cross-dataset metric only if relationship and grain are safe.

    Returns (metric_dict_or_none, list_of_safety_warnings).
    """
    safety_warnings = []
    source_id = source_meta.get("id", "source")
    source_name = source_meta.get("name", "Source")
    target_id = target_meta.get("id", "target")
    target_name = target_meta.get("name", "Target")

    source_col = rel.get("source_column")
    target_col = rel.get("target_column")
    cardinality = rel.get("cardinality")
    is_ambiguous = rel.get("is_ambiguous", False)
    val_status = rel.get("validation_status", "candidate")
    fan_out = rel.get("fan_out_ratio", 1.0)
    has_double_counting = rel.get("double_counting_risk", False)

    # 0. Validation status check: only explicitly validated relationships may be used for cross-dataset metrics
    if val_status != "validated":
        if is_ambiguous:
            safety_warnings.append({
                "source_dataset_id": source_id,
                "target_dataset_id": target_id,
                "warning_type": "ambiguous_relationship",
                "message": f"Join between '{source_name}.{source_col}' and '{target_name}.{target_col}' is ambiguous with competing candidates. Cross-dataset metrics excluded.",
                "risk": "ambiguity_risk",
                "affected_measures": rel.get("double_counting_measures", []),
            })
        elif val_status == "rejected":
            safety_warnings.append({
                "source_dataset_id": source_id,
                "target_dataset_id": target_id,
                "warning_type": "rejected_relationship",
                "message": f"Join between '{source_name}.{source_col}' and '{target_name}.{target_col}' was rejected ({rel.get('rejection_reason') or 'unmet join constraints'}). Cross-dataset metrics excluded.",
                "risk": "rejected_join_risk",
                "affected_measures": rel.get("double_counting_measures", []),
            })
        elif has_double_counting or fan_out > 1.05:
            safety_warnings.append({
                "source_dataset_id": source_id,
                "target_dataset_id": target_id,
                "warning_type": "fan_out",
                "message": f"Join between '{source_name}' and '{target_name}' exhibits fan-out multiplier of {fan_out:.2f}x (cardinality: {cardinality}). Unaggregated measures risk double counting.",
                "risk": "double_counting_risk",
                "affected_measures": rel.get("double_counting_measures", []),
            })
        return None, safety_warnings

    # 1. Ambiguity check: do NOT silently proceed if ambiguous
    if is_ambiguous:
        safety_warnings.append({
            "source_dataset_id": source_id,
            "target_dataset_id": target_id,
            "warning_type": "ambiguous_relationship",
            "message": f"Join between '{source_name}.{source_col}' and '{target_name}.{target_col}' is ambiguous with competing candidates. Cross-dataset metrics excluded.",
            "risk": "ambiguity_risk",
            "affected_measures": rel.get("double_counting_measures", []),
        })
        return None, safety_warnings

    # 2. Fan-out / Double-counting check: if fan_out > 1.0 or Many-to-Many
    if cardinality == "many_to_many" or fan_out > 1.05 or has_double_counting:
        safety_warnings.append({
            "source_dataset_id": source_id,
            "target_dataset_id": target_id,
            "warning_type": "fan_out",
            "message": f"Join between '{source_name}' and '{target_name}' exhibits fan-out multiplier of {fan_out:.2f}x (cardinality: {cardinality}). Unaggregated measures risk double counting.",
            "risk": "double_counting_risk",
            "affected_measures": rel.get("double_counting_measures", []),
        })
        # If fan out is severe (>1.5x) or many-to-many, we reject direct join computation
        if cardinality == "many_to_many" or fan_out > 1.5:
            return None, safety_warnings

    # Safe to compute: look for a measure in one dataset and a dimension in the other
    s_measures = [
        c["name"] for c in source_meta.get("columns", [])
        if (c.get("role") == "measure" or pd.api.types.is_numeric_dtype(source_df[c["name"]]))
        and c["name"] in source_df.columns
        and c["name"] != source_col
    ]
    t_dims = [
        c["name"] for c in target_meta.get("columns", [])
        if (c.get("role") in ("dimension", "categorical") or str(target_df[c["name"]].dtype) == "object")
        and c["name"] in target_df.columns
        and c["name"] != target_col
    ]

    t_measures = [
        c["name"] for c in target_meta.get("columns", [])
        if (c.get("role") == "measure" or pd.api.types.is_numeric_dtype(target_df[c["name"]]))
        and c["name"] in target_df.columns
        and c["name"] != target_col
    ]
    s_dims = [
        c["name"] for c in source_meta.get("columns", [])
        if (c.get("role") in ("dimension", "categorical") or str(source_df[c["name"]].dtype) == "object")
        and c["name"] in source_df.columns
        and c["name"] != source_col
    ]

    if s_measures and t_dims:
        meas_df, meas_meta, meas_col, meas_key = source_df, source_meta, s_measures[0], source_col
        dim_df, dim_meta, dim_col, dim_key = target_df, target_meta, t_dims[0], target_col
    elif t_measures and s_dims:
        meas_df, meas_meta, meas_col, meas_key = target_df, target_meta, t_measures[0], target_col
        dim_df, dim_meta, dim_col, dim_key = source_df, source_meta, s_dims[0], source_col
    else:
        return None, safety_warnings

    meas_dataset_name = meas_meta.get("name", "Dataset 1")
    dim_dataset_name = dim_meta.get("name", "Dataset 2")

    # Perform safe inner/left merge on bounded samples
    s_sub = meas_df[[meas_key, meas_col]].dropna().copy()
    s_sub[meas_col] = pd.to_numeric(s_sub[meas_col], errors="coerce")
    t_sub = dim_df[[dim_key, dim_col]].dropna().drop_duplicates(subset=[dim_key]).copy()

    if s_sub.empty or t_sub.empty:
        return None, safety_warnings

    merged = pd.merge(s_sub, t_sub, left_on=meas_key, right_on=dim_key, how="inner")
    if merged.empty:
        return None, safety_warnings

    grouped = merged.groupby(dim_col)[meas_col].agg(["sum", "count"]).reset_index()
    grouped = grouped.sort_values(by="sum", ascending=False)
    total_val = grouped["sum"].sum()

    top_breakdown = []
    for _, row in grouped.head(6).iterrows():
        top_breakdown.append({
            "group": str(row[dim_col])[:40],
            "sum": safe_float(row["sum"]),
            "count": int(row["count"]),
            "share_pct": safe_float((row["sum"] / total_val * 100.0) if total_val and total_val > 0 else 0.0),
        })

    metric = {
        "id": f"metric_cross_{meas_meta.get('id', 'm')}_{meas_col}_joined_{dim_meta.get('id', 'd')}_{dim_col}",
        "label": f"Cross-Dataset: {meas_col} by {dim_dataset_name} {dim_col}",
        "category": "cross_dataset",
        "dataset_id": meas_meta.get("id", "cross"),
        "dataset_name": f"{meas_dataset_name} ⋈ {dim_dataset_name}",
        "column_name": meas_col,
        "secondary_column": dim_col,
        "aggregation": "cross_sum_by_group",
        "status": "available",
        "reason_unavailable": None,
        "value": safe_float(total_val),
        "summary_stats": {
            "source_dataset": meas_dataset_name,
            "target_dataset": dim_dataset_name,
            "join_keys": f"{meas_key} = {dim_key}",
            "cardinality": cardinality,
            "joined_records": len(merged),
            "breakdown": top_breakdown,
        },
        "time_range": None,
        "grouping": {"dimension": dim_col, "dataset": dim_dataset_name},
        "evidence": [
            f"Successfully merged {len(merged):,} records along validated {cardinality} relationship",
            f"Aggregated '{meas_col}' from {meas_dataset_name} grouped by '{dim_col}' from {dim_dataset_name}",
        ],
        "caveats": [
            "Cross-dataset aggregation verified safe from grain fan-out",
            f"Target keys deduplicated to prevent measure inflation (fan-out ratio: {fan_out:.2f}x)",
        ],
        "warnings": [],
    }

    return metric, safety_warnings


def generate_explainable_insights(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Synthesize descriptive, evidence-backed insights from computed metrics.

    Strictly preserves non-causal language and explicit heuristic confidence.
    """
    insights = []

    for m in metrics:
        if m.get("status") != "available":
            continue

        cat = m.get("category")
        stats = m.get("summary_stats", {})

        # 1. Categorical Concentration Insight
        if cat in ("breakdown", "cross_dataset"):
            breakdown = stats.get("breakdown", [])
            if breakdown and len(breakdown) >= 2:
                top = breakdown[0]
                share = top.get("share_pct")
                if share is not None and share >= 40.0:
                    insights.append({
                        "id": f"insight_skew_{m['id']}",
                        "title": f"Concentration in {top['group']}",
                        "finding": f"'{top['group']}' represents {share:.1f}% of total {m.get('column_name', 'metric')} volume.",
                        "evidence": [
                            f"Observed in dataset '{m['dataset_name']}' across {stats.get('groups_count', len(breakdown))} categories",
                            f"Top segment accounts for {top.get('sum', 0):,} of total volume",
                        ],
                        "relevant_entity": None,
                        "relevant_dimension": m.get("secondary_column") or m.get("column_name"),
                        "relevant_time_period": None,
                        "why_it_matters": "High concentration in a single category can create operational reliance and variance risk.",
                        "confidence": 0.85,
                        "confidence_type": "heuristic",
                        "limitations": [
                            "Descriptive distribution observation only; does not prove customer or operational preference drivers.",
                            "External market factors or catalog selection may influence category representation.",
                        ],
                        "is_causal_claim": False,
                    })

        # 2. Temporal Trend Insight
        elif cat == "trend":
            change = stats.get("change_pct")
            direction = stats.get("direction")
            period_type = stats.get("period_type", "period")
            periods = stats.get("periods", [])
            if change is not None and abs(change) >= 5.0 and periods:
                verb = "increased" if change > 0 else "declined"
                insights.append({
                    "id": f"insight_trend_{m['id']}",
                    "title": f"{m.get('column_name', 'Metric')} {direction.capitalize()} Trend",
                    "finding": f"{m.get('column_name', 'Metric')} {verb} by {abs(change):.1f}% from {periods[0]['period']} to {periods[-1]['period']}.",
                    "evidence": [
                        f"Analyzed {len(periods)} consecutive {period_type} cycles in '{m['dataset_name']}'",
                        f"Initial value: {periods[0].get('value', 0):,}; final value: {periods[-1].get('value', 0):,}",
                    ],
                    "relevant_entity": None,
                    "relevant_dimension": None,
                    "relevant_time_period": f"{periods[0]['period']} to {periods[-1]['period']}",
                    "why_it_matters": f"Sustained {direction} shifts in volume influence capacity planning and baseline expectations.",
                    "confidence": 0.80,
                    "confidence_type": "heuristic",
                    "limitations": [
                        "Historical trend does not guarantee future trajectory.",
                        "Seasonality or reporting cycle irregularities may contribute to observed differences.",
                    ],
                    "is_causal_claim": False,
                })

        # 3. Top Performer Concentration
        elif cat == "top_performers":
            conc = stats.get("top_n_concentration_pct")
            top_performers = stats.get("top_performers", [])
            if conc is not None and conc >= 40.0 and top_performers:
                n = len(top_performers)
                insights.append({
                    "id": f"insight_performers_{m['id']}",
                    "title": f"Top {n} Entities Drive {conc:.1f}% of Volume",
                    "finding": f"The top {n} {m.get('secondary_column', 'entities')} account for {conc:.1f}% of cumulative {m.get('column_name', 'volume')}.",
                    "evidence": [
                        f"Measured across {stats.get('total_entities', 0):,} distinct entities in '{m['dataset_name']}'",
                        f"Leading entity: '{top_performers[0]['entity']}' ({top_performers[0].get('share_pct', 0):.1f}% share)",
                    ],
                    "relevant_entity": top_performers[0]["entity"],
                    "relevant_dimension": m.get("secondary_column"),
                    "relevant_time_period": None,
                    "why_it_matters": "A small cluster of entities drives a disproportionate share of business activity.",
                    "confidence": 0.88,
                    "confidence_type": "heuristic",
                    "limitations": [
                        "Identifies volume distribution only; does not reflect profitability or retention.",
                    ],
                    "is_causal_claim": False,
                })

        # 4. Summary / Outlier / Distribution Insight
        elif cat == "summary":
            skew = stats.get("skewness")
            mean_val = stats.get("mean")
            med_val = stats.get("median")
            if skew is not None and abs(skew) > 1.8 and mean_val and med_val:
                insights.append({
                    "id": f"insight_dist_{m['id']}",
                    "title": f"High Distribution Skew in {m.get('column_name')}",
                    "finding": f"The mean ({mean_val}) significantly diverges from the median ({med_val}), indicating right-tail extreme values.",
                    "evidence": [
                        f"Skewness coefficient of {skew:.2f} across {stats.get('count', 0):,} observations",
                        f"Max value ({stats.get('max')}) is multiple times the median ({med_val})",
                    ],
                    "relevant_entity": None,
                    "relevant_dimension": m.get("column_name"),
                    "relevant_time_period": None,
                    "why_it_matters": "When distributions are heavily skewed, decision thresholds based on averages can distort individual case expectations.",
                    "confidence": 0.82,
                    "confidence_type": "heuristic",
                    "limitations": [
                        "Statistical measure only; requires domain validation to determine whether outliers represent data entry errors or high-value events.",
                    ],
                    "is_causal_claim": False,
                })

    # Return top 5 unique insights
    return insights[:5]


def surface_investigation_opportunities(
    metrics: List[Dict[str, Any]],
    insights: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Surface actionable investigation opportunities linked directly to metrics and insights.

    Never invents ROI, savings, or guaranteed actions.
    """
    opportunities = []

    for ins in insights:
        linked_metric_id = ins["id"].replace("insight_skew_", "").replace("insight_trend_", "").replace("insight_performers_", "").replace("insight_dist_", "")

        if "Concentration" in ins["title"]:
            opportunities.append({
                "id": f"opp_investigate_skew_{ins['id']}",
                "title": f"Investigate Category Concentration in {ins.get('relevant_dimension', 'Dimension')}",
                "candidate_area": f"Examine dependency on dominant category '{ins.get('title').replace('Concentration in ', '')}' and identify whether secondary categories have growth headroom.",
                "linked_metric_id": linked_metric_id,
                "linked_insight_id": ins["id"],
                "evidence": ins["evidence"],
                "missing_evidence": [
                    "Margin or cost breakdown by category to assess net financial contribution",
                    "Customer demographic or channel attributes to explain category adoption",
                ],
                "uncertainty_notes": [
                    "This is an exploratory investigation candidate. Do not alter allocation or marketing mix without margin confirmation.",
                ],
                "action_type": "investigation_candidate",
            })
        elif "Trend" in ins["title"]:
            opportunities.append({
                "id": f"opp_investigate_trend_{ins['id']}",
                "title": f"Evaluate Drivers of {ins.get('relevant_time_period', 'Period')} Trajectory",
                "candidate_area": f"Audit external events, inventory availability, or campaign cycles during {ins.get('relevant_time_period', 'the observed timeframe')}.",
                "linked_metric_id": linked_metric_id,
                "linked_insight_id": ins["id"],
                "evidence": ins["evidence"],
                "missing_evidence": [
                    "Granular event logs or operational changes during the inflection period",
                    "Seasonality baseline from preceding years",
                ],
                "uncertainty_notes": [
                    "Historical movement may be seasonal or cyclical. Baseline comparison required before establishing operational changes.",
                ],
                "action_type": "investigation_candidate",
            })
        elif "Top" in ins["title"] and "Entities" in ins["title"]:
            opportunities.append({
                "id": f"opp_investigate_key_accounts_{ins['id']}",
                "title": f"Review Health and Retention of Key {ins.get('relevant_dimension', 'Entities')}",
                "candidate_area": "Prioritize account reviews or service agreements for high-volume entities to mitigate concentration loss risks.",
                "linked_metric_id": linked_metric_id,
                "linked_insight_id": ins["id"],
                "evidence": ins["evidence"],
                "missing_evidence": [
                    "Recent engagement, support ticket history, or contract renewal schedules",
                    "Profitability and servicing cost per top entity",
                ],
                "uncertainty_notes": [
                    "High volume does not always equate to high retention risk or high profitability.",
                ],
                "action_type": "investigation_candidate",
            })
        elif "Skew" in ins["title"]:
            opportunities.append({
                "id": f"opp_investigate_outliers_{ins['id']}",
                "title": f"Audit Outlier Distribution in {ins.get('relevant_dimension', 'Measure')}",
                "candidate_area": "Verify whether high-value tail records represent strategic VIP transactions or data capture anomalies.",
                "linked_metric_id": linked_metric_id,
                "linked_insight_id": ins["id"],
                "evidence": ins["evidence"],
                "missing_evidence": [
                    "Source transaction logs or validation against billing/ledger records",
                ],
                "uncertainty_notes": [
                    "Extreme values can distort decision models if unnormalized.",
                ],
                "action_type": "investigation_candidate",
            })

    # Deduplicate and return top 4 opportunities
    seen = set()
    unique_opps = []
    for opp in opportunities:
        if opp["title"] not in seen:
            seen.add(opp["title"])
            unique_opps.append(opp)

    return unique_opps[:4]
