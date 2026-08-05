"""Adversarial tests for deterministic dataset profiler.

Covers every hygiene bug class hit on Telco / Superstore / HR — as synthetic
columns, not those datasets themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.dataset_profiler import (
    NUMERIC_PARSE_THRESHOLD,
    ProfilingError,
    apply_feature_exclusions,
    profile_dataframe,
)


def _col(report, name: str):
    for c in report.columns:
        if c.column_name == name:
            return c
    raise AssertionError(f"column {name!r} missing from profile")


def test_mostly_numeric_with_5pct_garbage_still_numeric():
    n = 100
    vals = [str(1000 + i) for i in range(95)] + ["N/A", "oops", "xx", "??", "bad"]
    df = pd.DataFrame({"almost_num": vals[:n]})
    report = profile_dataframe(df)
    col = _col(report, "almost_num")
    assert col.inferred_type == "numeric"
    assert col.action_taken == "keep_as_numeric"
    assert col.numeric_parse_frac >= NUMERIC_PARSE_THRESHOLD
    assert "partial_numeric_parse_failures" in col.issues or col.failed_parse_samples


def test_fully_numeric_looking_strings_coerce():
    df = pd.DataFrame({"Store_Sales": ["1200", "3400", "5600", "7800", "116320"] * 4})
    report = profile_dataframe(df)
    col = _col(report, "Store_Sales")
    assert col.inferred_type == "numeric"
    assert col.action_taken == "keep_as_numeric"
    assert col.numeric_parse_frac == 1.0


def test_plain_id_column_excluded():
    df = pd.DataFrame({"id": list(range(50)), "x": np.random.randn(50)})
    report = profile_dataframe(df)
    col = _col(report, "id")
    assert col.inferred_type == "id"
    assert col.action_taken == "exclude_as_id"
    assert "id" in report.excluded_as_id


@pytest.mark.parametrize(
    "name",
    [" Store ID ", "RefNo", "employee_number", "pk_customer", "EmployeeNumber"],
)
def test_unusual_id_names_excluded(name):
    df = pd.DataFrame({name: list(range(40)), "feat": np.random.randn(40)})
    report = profile_dataframe(df)
    col = _col(report, name)
    assert col.inferred_type == "id", col
    assert col.action_taken == "exclude_as_id"


def test_near_unique_integer_without_id_name_still_excluded():
    """0-based contiguous integers with no ID substring → behavioral surrogate key."""
    df = pd.DataFrame(
        {
            "seq_token": list(range(100)),
            "y": [0, 1] * 50,
        }
    )
    report = profile_dataframe(df)
    col = _col(report, "seq_token")
    assert col.inferred_type == "id"
    assert col.action_taken == "exclude_as_id"


def test_near_unique_sales_not_treated_as_id():
    """Near-unique numeric sales must stay a feature (Superstore case)."""
    sales = [str(10000 + i * 37) for i in range(100)]
    df = pd.DataFrame({"Store_Sales": sales, "y": [0, 1] * 50})
    report = profile_dataframe(df)
    col = _col(report, "Store_Sales")
    assert col.inferred_type == "numeric"
    assert col.action_taken == "keep_as_numeric"
    assert "Store_Sales" not in report.excluded_as_id


def test_constant_column_dropped():
    df = pd.DataFrame(
        {
            "EmployeeCount": [1] * 30,
            "Over18": ["Y"] * 30,
            "age": [20 + (i % 12) for i in range(30)],
        }
    )
    report = profile_dataframe(df)
    assert _col(report, "EmployeeCount").action_taken == "drop_constant"
    assert _col(report, "Over18").action_taken == "drop_constant"
    assert "EmployeeCount" in report.dropped_as_constant
    cleaned = apply_feature_exclusions(
        ["EmployeeCount", "Over18", "age"], report
    )
    assert cleaned == ["age"]


def test_positive_label_missing_blocks():
    df = pd.DataFrame(
        {
            "Attrition": ["Yes", "No"] * 20,
            "Age": [22 + (i % 18) for i in range(40)],
        }
    )
    report = profile_dataframe(
        df,
        target_column="Attrition",
        target_positive_label="1",
        problem_type="binary_classification",
        feature_columns=["Age"],
    )
    assert report.blocking is True
    assert any(i["code"] == "positive_label_not_in_data" for i in report.blocking_issues)
    assert "Yes" in (report.present_target_values or [])
    with pytest.raises(ProfilingError) as ei:
        report.raise_if_blocking()
    detail = ei.value.as_detail()
    assert detail["code"] == "dataset_profiling_blocked"
    assert "present_target_values" in detail


def test_positive_label_ok_does_not_block():
    df = pd.DataFrame(
        {
            "Attrition": ["Yes", "No"] * 20,
            "Age": [22 + (i % 18) for i in range(40)],
        }
    )
    report = profile_dataframe(
        df,
        target_column="Attrition",
        target_positive_label="Yes",
        feature_columns=["Age"],
    )
    assert report.blocking is False


def test_multiclass_forced_binary_warns():
    df = pd.DataFrame(
        {
            "Performance": ["High", "Medium", "Low"] * 20,
            "sales": [str(1000 + i * 17) for i in range(60)],
        }
    )
    report = profile_dataframe(
        df,
        target_column="Performance",
        target_positive_label="High",
        problem_type="binary_classification",
        feature_columns=["sales"],
    )
    assert report.blocking is False
    assert any(w["code"] == "multiclass_forced_binary" for w in report.warnings)


def test_high_missing_density_warns():
    # 40% missing; present values are non-sequential measurements
    vals: list = [100.0 + (i % 17) * 3.5 for i in range(60)] + [np.nan] * 40
    df = pd.DataFrame({"sparse": vals, "ok": np.random.randn(100)})
    report = profile_dataframe(df)
    col = _col(report, "sparse")
    assert "high_missing_density" in col.issues
    assert any(w["code"] == "high_missing_density" for w in report.warnings)
    assert col.inferred_type == "numeric"


def test_high_cardinality_categorical_warns():
    notes = [f"note about topic {i % 17} variant {i}" for i in range(80)]
    df = pd.DataFrame({"free_text": notes, "y": [0, 1] * 40})
    report = profile_dataframe(df)
    col = _col(report, "free_text")
    assert col.inferred_type == "categorical"
    assert col.action_taken == "keep_as_categorical"
    assert "high_cardinality_categorical" in col.issues
    assert any(w["code"] == "high_cardinality_categorical" for w in report.warnings)


def test_string_categorical_not_zeroed_as_numeric():
    """Regression: pandas dtype=str Gender/OverTime must stay categorical."""
    df = pd.DataFrame(
        {
            "Gender": ["Female", "Male"] * 25,
            "OverTime": ["Yes", "No"] * 25,
            "MonthlyIncome": [3000 + (i * 37) % 5000 for i in range(50)],
        }
    )
    df["Gender"] = df["Gender"].astype("string")
    df["OverTime"] = df["OverTime"].astype("string")
    report = profile_dataframe(df)
    assert _col(report, "Gender").inferred_type == "categorical"
    assert _col(report, "OverTime").inferred_type == "categorical"
    assert _col(report, "MonthlyIncome").inferred_type == "numeric"


def test_profiling_error_detail_shape():
    err = ProfilingError(
        "blocked",
        code="positive_label_not_in_data",
        blocking_issues=[{"code": "positive_label_not_in_data"}],
        present_target_values=["Yes", "No"],
    )
    d = err.as_detail()
    assert d["code"] == "positive_label_not_in_data"
    assert d["message"] == "blocked"
    assert d["present_target_values"] == ["Yes", "No"]


def test_all_features_excluded_blocks_when_requested():
    df = pd.DataFrame(
        {
            "id": list(range(20)),
            "EmployeeCount": [1] * 20,
            "target": ["Yes", "No"] * 10,
        }
    )
    report = profile_dataframe(
        df,
        target_column="target",
        target_positive_label="Yes",
        feature_columns=["id", "EmployeeCount"],
    )
    assert report.blocking is True
    assert any(i["code"] == "no_usable_features" for i in report.blocking_issues)
