"""Domain-aware recommendation catalogs (Telco vs HR attrition)."""

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
