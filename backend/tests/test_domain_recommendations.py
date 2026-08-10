"""Domain-aware recommendation catalogs (Telco vs HR attrition)."""

from __future__ import annotations

import unittest

from app.recommendations.action_catalog import (
    get_applicable_actions,
    get_catalog,
)
from app.recommendations.decision_scorer import DecisionScorer
from app.recommendations.domains import (
    DOMAIN_HR_ATTRITION,
    DOMAIN_TELCO,
    detect_domain,
)


def test_detect_domain_hr_by_name():
    assert (
        detect_domain(project_name="IBM Employee Attrition", target_column="Attrition")
        == DOMAIN_HR_ATTRITION
    )


def test_detect_domain_telco_by_name():
    assert (
        detect_domain(project_name="Telco Churn Test", target_column="Churn")
        == DOMAIN_TELCO
    )


def test_detect_domain_hr_by_features():
    cols = [
        "OverTime",
        "JobSatisfaction",
        "MonthlyIncome",
        "YearsAtCompany",
        "WorkLifeBalance",
        "DistanceFromHome",
    ]
    assert detect_domain(feature_columns=cols) == DOMAIN_HR_ATTRITION


def test_detect_domain_telco_by_features():
    cols = ["tenure", "MonthlyCharges", "Contract", "TechSupport", "InternetService"]
    assert detect_domain(feature_columns=cols) == DOMAIN_TELCO


def test_hr_catalog_has_no_telco_discounts():
    codes = set(get_catalog(DOMAIN_HR_ATTRITION).keys())
    assert "discount_10" not in codes
    assert "reduce_overtime" in codes
    assert "manager_stay_conversation" in codes


def test_telco_catalog_unchanged_default():
    codes = set(get_catalog(DOMAIN_TELCO).keys())
    assert "discount_10" in codes
    assert "reduce_overtime" not in codes


def test_hr_scoring_prefers_hr_actions():
    features = {
        "OverTime": "Yes",
        "JobSatisfaction": 1,
        "MonthlyIncome": 2800,
        "YearsSinceLastPromotion": 4,
        "DistanceFromHome": 25,
        "WorkLifeBalance": 1,
        "BusinessTravel": "Travel_Frequently",
        "StockOptionLevel": 0,
    }
    drivers = [
        {"feature": "OverTime", "impact": 0.4},
        {"feature": "JobSatisfaction", "impact": 0.25},
        {"feature": "YearsSinceLastPromotion", "impact": 0.15},
    ]
    result = DecisionScorer().score_case(
        features=features,
        probability=0.62,
        top_factors=drivers,
        domain=DOMAIN_HR_ATTRITION,
        top_n=5,
    )
    codes = [r["action_code"] for r in result["recommendations"]]
    assert codes, "expected HR recommendations"
    assert all(c not in ("discount_10", "discount_20", "upgrade_contract_1yr") for c in codes)
    assert any(
        c in ("reduce_overtime", "manager_stay_conversation", "promotion_path_plan")
        for c in codes
    )
    assert result["scoring"]["domain"] == DOMAIN_HR_ATTRITION


def test_telco_scoring_still_returns_telco_actions():
    features = {
        "Contract": "Month-to-month",
        "tenure": 8,
        "MonthlyCharges": 85.0,
        "TotalCharges": 900.0,
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
    }
    drivers = [
        {"feature": "Contract", "impact": 0.35},
        {"feature": "MonthlyCharges", "impact": 0.2},
    ]
    result = DecisionScorer().score_case(
        features=features,
        probability=0.7,
        top_factors=drivers,
        domain=DOMAIN_TELCO,
        top_n=5,
    )
    codes = [r["action_code"] for r in result["recommendations"]]
    assert any(c.startswith("discount") or "contract" in c or c == "plan_optimization" for c in codes)
    assert "reduce_overtime" not in codes


def test_hr_overtime_action_applicable():
    actions = get_applicable_actions(
        {"OverTime": "Yes", "JobSatisfaction": 2},
        0.5,
        domain=DOMAIN_HR_ATTRITION,
    )
    codes = {a.code for a in actions}
    assert "reduce_overtime" in codes


def test_learning_blend_requires_min_n():
    from app.recommendations.impact_calculator import ImpactCalculator
    from app.recommendations.action_catalog import get_action

    action = get_action("reduce_overtime", domain=DOMAIN_HR_ATTRITION)
    assert action is not None
    cold = ImpactCalculator(
        {"reduce_overtime": {"n": 2, "success_rate": 0.9, "reliable": False}}
    ).calculate_impact(action, {"OverTime": "Yes"}, 0.7)
    assert cold["learning"]["applied"] is False
    assert "need 3+" in (cold["learning"]["learning_note"] or "")

    hot = ImpactCalculator(
        {
            "reduce_overtime": {
                "n": 5,
                "success_n": 5,
                "success_rate": 1.0,
                "reliable": True,
            }
        }
    ).calculate_impact(action, {"OverTime": "Yes"}, 0.7)
    assert hot["learning"]["applied"] is True
    assert hot["components"]["learning_applied"] is True
    # Learned impact should pull toward historical success vs pure catalog
    base = ImpactCalculator().calculate_impact(action, {"OverTime": "Yes"}, 0.7)
    assert hot["impact_score"] >= base["impact_score"] - 1e-6


def test_zero_success_rate_tempers_not_ignored():
    """0.0 success_rate must not be treated as missing (falsy) data."""
    from app.recommendations.impact_calculator import ImpactCalculator
    from app.recommendations.action_catalog import get_action

    action = get_action("reduce_overtime", domain=DOMAIN_HR_ATTRITION)
    assert action is not None
    base = DecisionScorer().score_action(
        action,
        features={"OverTime": "Yes", "JobSatisfaction": 1},
        probability=0.62,
        feature_importance={"OverTime": 0.4},
        outcome_label="attrition",
    )
    zero = DecisionScorer(
        effectiveness_data={
            "reduce_overtime": {
                "n": 4,
                "success_n": 0,
                "success_rate": 0.0,
                "effectiveness_rate": 0.0,
                "reliable": True,
            }
        }
    ).score_action(
        action,
        features={"OverTime": "Yes", "JobSatisfaction": 1},
        probability=0.62,
        feature_importance={"OverTime": 0.4},
        outcome_label="attrition",
    )
    assert zero["learning_applied"] is True
    assert zero["effectiveness_rate"] == 0.0
    assert "tempered" in (zero.get("learning_note") or "").lower()
    assert zero["final_score"] <= base["final_score"] - 0.02

    # success_n/n without explicit rate still learns
    derived = ImpactCalculator(
        {"reduce_overtime": {"n": 4, "success_n": 0, "reliable": True}}
    ).calculate_impact(action, {"OverTime": "Yes"}, 0.7)
    assert derived["learning"]["applied"] is True
    assert derived["learning"]["effectiveness_rate"] == 0.0


def test_learning_boosts_high_success_action_in_ranking():
    features = {
        "OverTime": "Yes",
        "JobSatisfaction": 1,
        "MonthlyIncome": 2800,
        "YearsSinceLastPromotion": 4,
        "DistanceFromHome": 25,
        "WorkLifeBalance": 1,
        "BusinessTravel": "Travel_Frequently",
        "StockOptionLevel": 0,
    }
    drivers = [
        {"feature": "OverTime", "impact": 0.4},
        {"feature": "JobSatisfaction", "impact": 0.25},
    ]
    baseline = DecisionScorer().score_case(
        features=features,
        probability=0.62,
        top_factors=drivers,
        domain=DOMAIN_HR_ATTRITION,
        top_n=5,
    )
    learned = DecisionScorer(
        effectiveness_data={
            "reduce_overtime": {
                "n": 8,
                "n_outcomes": 8,
                "success_n": 7,
                "success_rate": 0.875,
                "effectiveness_rate": 0.875,
                "reliable": True,
            }
        }
    ).score_case(
        features=features,
        probability=0.62,
        top_factors=drivers,
        domain=DOMAIN_HR_ATTRITION,
        top_n=5,
    )
    row = next(
        (r for r in learned["recommendations"] if r["action_code"] == "reduce_overtime"),
        None,
    )
    assert row is not None
    assert row["learning_applied"] is True
    assert row["n_outcomes"] == 8
    assert "boosted" in (row.get("learning_note") or "").lower() or "favorable" in (
        row.get("learning_note") or ""
    ).lower()
    base_row = next(
        (r for r in baseline["recommendations"] if r["action_code"] == "reduce_overtime"),
        None,
    )
    if base_row:
        assert row["final_score"] >= base_row["final_score"] - 1e-6


def load_tests(loader, tests, pattern):
    """Expose pytest-style functions to unittest runners."""
    suite = unittest.TestSuite()
    g = globals()
    for name in sorted(g):
        if name.startswith("test_") and callable(g[name]):
            suite.addTest(unittest.FunctionTestCase(g[name]))
    return suite


if __name__ == "__main__":
    unittest.main()
