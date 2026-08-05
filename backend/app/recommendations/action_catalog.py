"""Action catalogs: retention / decision actions with applicability rules.

Telco churn is the default catalog. HR attrition (and future domains) use
separate catalogs selected via ``domains.detect_domain`` — never a global
override of Telco actions.

Feature keys are resolved with flexible aliases so Telco-style names
(Contract, MonthlyCharges) and HR names (OverTime, JobSatisfaction) match
template conditions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.recommendations.domains import (
    DEFAULT_DOMAIN,
    DOMAIN_HR_ATTRITION,
    DOMAIN_TELCO,
)


def _norm(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    s = s.lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", s).strip("_")


# Common raw column aliases → logical keys used in conditions / targets
_FEATURE_ALIASES: dict[str, list[str]] = {
    "contract_type": ["contract", "contracttype", "plan_term", "subscription_term"],
    "monthly_charges": ["monthlycharges", "monthly_charge", "bill_amount", "mrr"],
    "total_charges": ["totalcharges", "lifetime_value", "ltv", "spend"],
    "tenure": ["customer_tenure", "months_as_customer", "account_age_months"],
    "tech_support": ["techsupport"],
    "online_security": ["onlinesecurity"],
    "streaming_tv": ["streamingtv"],
    "streaming_movies": ["streamingmovies"],
    "payment_method": ["paymentmethod", "pay_method"],
    "paperless_billing": ["paperlessbilling"],
    "internet_service": ["internetservice"],
    "satisfaction_score": ["nps", "csat", "satisfaction"],
    "num_complaints": ["complaints"],
    "days_since_last_interaction": ["recency", "days_since_last_login", "last_contact_days"],
    "logins": ["login_count", "sessions", "app_usage"],
    # HR attrition (IBM Employee Attrition-style)
    "over_time": ["overtime", "ot"],
    "job_satisfaction": ["jobsatisfaction"],
    "environment_satisfaction": ["environmentsatisfaction"],
    "relationship_satisfaction": ["relationshipsatisfaction"],
    "work_life_balance": ["worklifebalance", "wlb"],
    "monthly_income": ["monthlyincome", "salary", "compensation"],
    "years_at_company": ["yearsatcompany", "company_tenure"],
    "years_in_current_role": ["yearsincurrentrole"],
    "years_since_last_promotion": ["yearssincelastpromotion"],
    "years_with_curr_manager": ["yearswithcurrmanager"],
    "distance_from_home": ["distancefromhome", "commute_distance"],
    "business_travel": ["businesstravel"],
    "job_involvement": ["jobinvolvement"],
    "job_level": ["joblevel"],
    "training_times_last_year": ["trainingtimeslastyear", "training"],
    "percent_salary_hike": ["percentsalaryhike", "salary_hike"],
    "stock_option_level": ["stockoptionlevel"],
    "num_companies_worked": ["numcompaniesworked"],
    "performance_rating": ["performancerating"],
    "total_working_years": ["totalworkingyears"],
}


def feature_lookup(features: dict[str, Any], logical_key: str, default: Any = None) -> Any:
    """Read a feature by logical name or any known alias / case form."""
    if not features:
        return default
    # exact
    if logical_key in features:
        return features[logical_key]
    n = _norm(logical_key)
    aliases = [_norm(a) for a in _FEATURE_ALIASES.get(logical_key, [])] + [n]
    # reverse map: any feature key that normalizes to logical or alias
    for k, v in features.items():
        kn = _norm(k)
        if kn == n or kn in aliases or n in kn or kn in n:
            return v
    return default


def build_feature_importance(
    top_factors: Optional[list[dict[str, Any]]],
) -> dict[str, float]:
    """Normalize Phase-2/3 drivers into signed importance keyed by logical + raw names."""
    imp: dict[str, float] = {}
    if not top_factors:
        return imp
    for f in top_factors:
        name = f.get("feature")
        if not name:
            continue
        raw = float(f.get("impact", f.get("shap_value", f.get("importance", 0)) or 0))
        imp[name] = raw
        imp[_norm(name)] = raw
        # Also map to logical keys when possible
        nk = _norm(name)
        for logical, aliases in _FEATURE_ALIASES.items():
            if nk == logical or nk in [_norm(a) for a in aliases] or logical in nk:
                # Prefer magnitude if already higher risk contribution (positive)
                if logical not in imp or abs(raw) > abs(imp[logical]):
                    imp[logical] = raw
    return imp


@dataclass
class Action:
    """Business action definition."""

    code: str
    name: str
    description: str
    base_cost: float  # 0-1 normalized
    impact_potential: float  # 0-1 expected impact
    implementation_time: str  # immediate, short, medium, long
    applicable_conditions: dict[str, Any] = field(default_factory=dict)
    target_features: list[str] = field(default_factory=list)
    category: str = "retention"  # retention | save | growth | service
    # Soft applicability: if True, failure of non-critical feature matches soft-excludes
    soft_match: bool = True


ACTION_CATALOG: dict[str, Action] = {
    "discount_10": Action(
        code="discount_10",
        name="10% discount offer",
        description="Offer ~10% off for a short retention window to reduce cost pressure.",
        base_cost=0.3,
        impact_potential=0.5,
        implementation_time="immediate",
        applicable_conditions={"min_probability": 0.35},
        target_features=["monthly_charges"],
        category="save",
    ),
    "discount_20": Action(
        code="discount_20",
        name="20% discount offer",
        description="Stronger short-term discount when risk is elevated and price is a driver.",
        base_cost=0.5,
        impact_potential=0.7,
        implementation_time="immediate",
        applicable_conditions={"min_probability": 0.55},
        target_features=["monthly_charges"],
        category="save",
    ),
    "plan_optimization": Action(
        code="plan_optimization",
        name="Plan fit review",
        description="Right-size the plan / remove unused paid extras to improve perceived value.",
        base_cost=0.12,
        impact_potential=0.45,
        implementation_time="short",
        applicable_conditions={"min_monthly_charges": 40},
        target_features=["monthly_charges"],
        category="save",
    ),
    "upgrade_contract_1yr": Action(
        code="upgrade_contract_1yr",
        name="1-year commitment offer",
        description="Trade fair value for longer term when currently flexible.",
        base_cost=0.35,
        impact_potential=0.6,
        implementation_time="short",
        applicable_conditions={
            "contract_type_in": ["month-to-month", "month_to_month", "Month-to-month"],
            "min_tenure": 2,
        },
        target_features=["contract_type"],
        category="save",
    ),
    "upgrade_contract_2yr": Action(
        code="upgrade_contract_2yr",
        name="2-year commitment offer",
        description="Longer commitment with stronger incentives for higher-risk flexible accounts.",
        base_cost=0.55,
        impact_potential=0.75,
        implementation_time="short",
        applicable_conditions={
            "contract_type_in": [
                "month-to-month",
                "month_to_month",
                "Month-to-month",
                "one year",
                "One year",
                "one_year",
            ],
            "min_tenure": 3,
            "min_probability": 0.4,
        },
        target_features=["contract_type"],
        category="save",
    ),
    "add_tech_support": Action(
        code="add_tech_support",
        name="Tech support trial",
        description="Trial tech support when support gaps may be creating silent friction.",
        base_cost=0.22,
        impact_potential=0.4,
        implementation_time="immediate",
        applicable_conditions={"tech_support_in": ["no", "No", "false", "0"]},
        target_features=["tech_support"],
        category="service",
    ),
    "add_security": Action(
        code="add_security",
        name="Security add-on trial",
        description="Trial online security if not already attached.",
        base_cost=0.22,
        impact_potential=0.35,
        implementation_time="immediate",
        applicable_conditions={"online_security_in": ["no", "No", "false", "0"]},
        target_features=["online_security"],
        category="service",
    ),
    "autopay_incentive": Action(
        code="autopay_incentive",
        name="Auto-pay incentive",
        description="Nudge easier billing to reduce friction-driven drop-off.",
        base_cost=0.15,
        impact_potential=0.35,
        implementation_time="immediate",
        applicable_conditions={
            "payment_method_in": [
                "electronic check",
                "Electronic check",
                "mailed check",
                "Mailed check",
                "electronic_check",
                "mailed_check",
            ],
        },
        target_features=["payment_method"],
        category="service",
    ),
    "proactive_support_call": Action(
        code="proactive_support_call",
        name="Proactive success call",
        description="Human check-in from customer success when risk or friction is elevated.",
        base_cost=0.2,
        impact_potential=0.5,
        implementation_time="short",
        applicable_conditions={"min_probability": 0.3},
        target_features=["satisfaction_score", "num_complaints", "tech_support"],
        category="save",
    ),
    "engagement_campaign": Action(
        code="engagement_campaign",
        name="Re-engagement campaign",
        description="Targeted nudge when usage/recency looks weak.",
        base_cost=0.15,
        impact_potential=0.35,
        implementation_time="immediate",
        applicable_conditions={},  # soft applicability via drivers
        target_features=["logins", "days_since_last_interaction", "tenure"],
        category="retention",
    ),
    "loyalty_program": Action(
        code="loyalty_program",
        name="Loyalty recognition",
        description="Reward stickiness for longer-tenure customers without only using discounts.",
        base_cost=0.18,
        impact_potential=0.35,
        implementation_time="immediate",
        applicable_conditions={"min_tenure": 6},
        target_features=["tenure", "total_charges"],
        category="growth",
    ),
    "vip_upgrade": Action(
        code="vip_upgrade",
        name="Priority / VIP handling",
        description="Elevate service priority for high-lifetime-value relationships at risk.",
        base_cost=0.35,
        impact_potential=0.5,
        implementation_time="immediate",
        applicable_conditions={
            "min_total_charges": 800,
            "min_probability": 0.35,
        },
        target_features=["total_charges", "tenure"],
        category="save",
    ),
    "check_in_light": Action(
        code="check_in_light",
        name="Light check-in",
        description="Low-cost health check — useful when risk is moderate or uncertainty is soft.",
        base_cost=0.08,
        impact_potential=0.25,
        implementation_time="immediate",
        applicable_conditions={},
        target_features=[],
        category="retention",
    ),
    "monitor_only": Action(
        code="monitor_only",
        name="Monitor — no heavy intervention",
        description="Keep an eye on signals; deprioritize costly actions while risk is low.",
        base_cost=0.02,
        impact_potential=0.05,
        implementation_time="immediate",
        applicable_conditions={"max_probability": 0.45},
        target_features=[],
        category="retention",
    ),
}

# Backward-compatible alias: default / telco catalog
TELCO_ACTION_CATALOG = ACTION_CATALOG

HR_ATTRITION_CATALOG: dict[str, Action] = {
    "reduce_overtime": Action(
        code="reduce_overtime",
        name="Overtime / workload reset",
        description="Cap overtime and rebalance workload when hours are driving exit risk.",
        base_cost=0.28,
        impact_potential=0.7,
        implementation_time="short",
        applicable_conditions={
            "over_time_in": ["yes", "Yes", "y", "true", "1"],
            "min_probability": 0.25,
        },
        target_features=["over_time"],
        category="save",
    ),
    "manager_stay_conversation": Action(
        code="manager_stay_conversation",
        name="Manager stay conversation",
        description="Structured 1:1 with the manager focused on concerns, goals, and blockers.",
        base_cost=0.12,
        impact_potential=0.55,
        implementation_time="immediate",
        applicable_conditions={"min_probability": 0.3},
        target_features=[
            "job_satisfaction",
            "relationship_satisfaction",
            "years_with_curr_manager",
            "environment_satisfaction",
        ],
        category="save",
    ),
    "comp_market_review": Action(
        code="comp_market_review",
        name="Compensation market review",
        description="Review pay vs market and recent hike when income or raise signals risk.",
        base_cost=0.55,
        impact_potential=0.65,
        implementation_time="medium",
        applicable_conditions={"min_probability": 0.4},
        target_features=["monthly_income", "percent_salary_hike", "job_level"],
        category="save",
    ),
    "promotion_path_plan": Action(
        code="promotion_path_plan",
        name="Promotion / growth path plan",
        description="Clear next-role plan when stalled tenure-in-role or time since promotion is high.",
        base_cost=0.22,
        impact_potential=0.6,
        implementation_time="short",
        applicable_conditions={
            "min_years_since_last_promotion": 2,
            "min_probability": 0.3,
        },
        target_features=[
            "years_since_last_promotion",
            "years_in_current_role",
            "job_level",
        ],
        category="save",
    ),
    "work_life_balance_plan": Action(
        code="work_life_balance_plan",
        name="Work–life balance plan",
        description="Concrete WLB changes (schedule, coverage, PTO) when balance scores are weak.",
        base_cost=0.2,
        impact_potential=0.55,
        implementation_time="short",
        applicable_conditions={"max_work_life_balance": 2},
        target_features=["work_life_balance", "over_time"],
        category="service",
    ),
    "hybrid_commute_flexibility": Action(
        code="hybrid_commute_flexibility",
        name="Hybrid / commute flexibility",
        description="Remote or hybrid options when distance-from-home is a friction driver.",
        base_cost=0.25,
        impact_potential=0.5,
        implementation_time="short",
        applicable_conditions={"min_distance_from_home": 15},
        target_features=["distance_from_home"],
        category="service",
    ),
    "travel_load_reduction": Action(
        code="travel_load_reduction",
        name="Travel load reduction",
        description="Reduce frequent travel when BusinessTravel is elevating attrition risk.",
        base_cost=0.3,
        impact_potential=0.5,
        implementation_time="short",
        applicable_conditions={
            "business_travel_in": [
                "travel_frequently",
                "Travel_Frequently",
                "frequently",
            ],
        },
        target_features=["business_travel"],
        category="service",
    ),
    "learning_development": Action(
        code="learning_development",
        name="Learning & development boost",
        description="Fund training or mentorship when involvement or training volume looks thin.",
        base_cost=0.18,
        impact_potential=0.4,
        implementation_time="short",
        applicable_conditions={},
        target_features=["training_times_last_year", "job_involvement"],
        category="growth",
    ),
    "recognition_program": Action(
        code="recognition_program",
        name="Recognition & appreciation",
        description="Visible recognition when satisfaction / involvement scores are soft.",
        base_cost=0.1,
        impact_potential=0.35,
        implementation_time="immediate",
        applicable_conditions={"max_job_satisfaction": 2},
        target_features=["job_satisfaction", "environment_satisfaction"],
        category="retention",
    ),
    "equity_refresh_talk": Action(
        code="equity_refresh_talk",
        name="Equity / stock options conversation",
        description="Discuss equity refresh when stock option level is low for the role.",
        base_cost=0.4,
        impact_potential=0.45,
        implementation_time="medium",
        applicable_conditions={
            "max_stock_option_level": 0,
            "min_probability": 0.35,
        },
        target_features=["stock_option_level", "monthly_income"],
        category="save",
    ),
    "internal_mobility": Action(
        code="internal_mobility",
        name="Internal role mobility",
        description="Explore an internal move when role fit or department friction is high.",
        base_cost=0.35,
        impact_potential=0.55,
        implementation_time="medium",
        applicable_conditions={"min_probability": 0.45},
        target_features=["job_satisfaction", "job_involvement", "years_in_current_role"],
        category="save",
    ),
    "hr_check_in_light": Action(
        code="hr_check_in_light",
        name="Light HR / people check-in",
        description="Low-cost people-ops pulse — useful for moderate risk or soft confidence.",
        base_cost=0.08,
        impact_potential=0.25,
        implementation_time="immediate",
        applicable_conditions={},
        target_features=[],
        category="retention",
    ),
    "monitor_attrition": Action(
        code="monitor_attrition",
        name="Monitor — no heavy HR intervention",
        description="Watch signals; avoid costly interventions while attrition risk is low.",
        base_cost=0.02,
        impact_potential=0.05,
        implementation_time="immediate",
        applicable_conditions={"max_probability": 0.45},
        target_features=[],
        category="retention",
    ),
}

DOMAIN_CATALOGS: dict[str, dict[str, Action]] = {
    DOMAIN_TELCO: TELCO_ACTION_CATALOG,
    DOMAIN_HR_ATTRITION: HR_ATTRITION_CATALOG,
}


def get_catalog(domain: Optional[str] = None) -> dict[str, Action]:
    """Return action catalog for domain (defaults to Telco)."""
    key = domain or DEFAULT_DOMAIN
    return DOMAIN_CATALOGS.get(key) or TELCO_ACTION_CATALOG


def get_action(code: str, domain: Optional[str] = None) -> Optional[Action]:
    catalog = get_catalog(domain)
    if code in catalog:
        return catalog[code]
    for cat in DOMAIN_CATALOGS.values():
        if code in cat:
            return cat[code]
    return ACTION_CATALOG.get(code)


def get_all_actions(domain: Optional[str] = None) -> list[Action]:
    return list(get_catalog(domain).values())


def get_applicable_actions(
    features: dict[str, Any],
    probability: float,
    *,
    domain: Optional[str] = None,
    include_soft_defaults: bool = True,
) -> list[Action]:
    catalog = get_catalog(domain)
    applicable = []
    for action in catalog.values():
        ok, _ = check_conditions(action, features, probability)
        if ok:
            applicable.append(action)
    if not applicable and include_soft_defaults:
        soft_codes = (
            ("hr_check_in_light", "monitor_attrition", "manager_stay_conversation")
            if (domain or DEFAULT_DOMAIN) == DOMAIN_HR_ATTRITION
            else ("check_in_light", "monitor_only", "plan_optimization")
        )
        for code in soft_codes:
            a = catalog.get(code)
            if a and a not in applicable:
                applicable.append(a)
    return applicable


def check_conditions(
    action: Action,
    features: dict[str, Any],
    probability: float,
) -> tuple[bool, list[str]]:
    """Return (ok, fail_reasons)."""
    fails: list[str] = []
    conditions = action.applicable_conditions or {}
    p = float(probability)

    for condition, value in conditions.items():
        if condition == "min_probability" or condition == "min_churn_probability":
            if p < float(value):
                fails.append(f"probability<{value}")
        elif condition == "max_probability" or condition == "max_churn_probability":
            if p > float(value):
                fails.append(f"probability>{value}")
        elif condition == "min_tenure":
            if float(feature_lookup(features, "tenure", 0) or 0) < float(value):
                fails.append("tenure")
        elif condition == "min_total_charges":
            if float(feature_lookup(features, "total_charges", 0) or 0) < float(value):
                fails.append("total_charges")
        elif condition == "min_monthly_charges":
            if float(feature_lookup(features, "monthly_charges", 0) or 0) < float(value):
                fails.append("monthly_charges")
        elif condition == "max_satisfaction_score":
            sat = feature_lookup(features, "satisfaction_score", None)
            if sat is not None and float(sat) > float(value):
                fails.append("satisfaction_score")
        elif condition == "min_complaints":
            if float(feature_lookup(features, "num_complaints", 0) or 0) < float(value):
                fails.append("complaints")
        elif condition == "min_distance_from_home":
            if float(feature_lookup(features, "distance_from_home", 0) or 0) < float(value):
                fails.append("distance_from_home")
        elif condition == "max_work_life_balance":
            wlb = feature_lookup(features, "work_life_balance", None)
            if wlb is None or float(wlb) > float(value):
                fails.append("work_life_balance")
        elif condition == "max_job_satisfaction":
            js = feature_lookup(features, "job_satisfaction", None)
            if js is None or float(js) > float(value):
                fails.append("job_satisfaction")
        elif condition == "min_years_since_last_promotion":
            if float(feature_lookup(features, "years_since_last_promotion", 0) or 0) < float(
                value
            ):
                fails.append("years_since_last_promotion")
        elif condition == "max_stock_option_level":
            so = feature_lookup(features, "stock_option_level", None)
            if so is None or float(so) > float(value):
                fails.append("stock_option_level")
        elif condition.endswith("_in"):
            feat = condition[: -len("_in")]
            raw = feature_lookup(features, feat, None)
            if raw is None:
                if not action.soft_match:
                    fails.append(feat)
                continue
            allowed = [str(v).strip().lower() for v in value]
            if str(raw).strip().lower() not in allowed:
                fails.append(feat)
        elif condition in features or feature_lookup(features, condition, "__missing__") != "__missing__":
            feature_value = feature_lookup(features, condition)
            if isinstance(value, list):
                if str(feature_value).strip().lower() not in [str(v).lower() for v in value]:
                    fails.append(condition)
            elif feature_value != value:
                fails.append(condition)

    return (len(fails) == 0, fails)


def action_from_custom(
    *,
    code: str,
    name: str,
    description: str = "",
    estimated_cost: float = 0.0,
    estimated_impact: float = 0.5,
    applicable_when: Any = None,
) -> Action:
    """Map org CustomAction row into catalog Action shape."""
    # cost may be dollars — normalize lightly
    if estimated_cost is None:
        cost = 0.2
    elif estimated_cost > 1.5:
        cost = min(1.0, float(estimated_cost) / 500.0)
    else:
        cost = float(min(1.0, max(0.0, estimated_cost)))

    targets: list[str] = []
    if isinstance(applicable_when, dict):
        targets = list(applicable_when.keys())
    elif isinstance(applicable_when, str) and applicable_when:
        targets = [applicable_when]

    return Action(
        code=code,
        name=name,
        description=description or name,
        base_cost=cost,
        impact_potential=float(min(1.0, max(0.05, estimated_impact or 0.5))),
        implementation_time="short",
        applicable_conditions={},  # custom actions score even if unfiltered
        target_features=[_norm(t) for t in targets],
        category="custom",
        soft_match=True,
    )
