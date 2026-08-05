"""Feature → business concept mapping (domain-agnostic with rich aliases).

Maps many raw column names (including camelCase / Telco) onto reusable
categories so insight templates work without hardcoding a single dataset.
"""

from __future__ import annotations

import re
from typing import Any, Optional


def _normalize_key(name: str) -> str:
    s = str(name).strip()
    # camelCase / PascalCase → snake-ish
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = s.lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    # drop missing-indicator suffix noise
    s = re.sub(r"_missing$", "", s)
    s = re.sub(r"_is_missing$", "", s)
    return s


# Canonical concepts keyed by normalized name
FEATURE_MAPPING: dict[str, dict[str, Any]] = {
    # --- tenure / time ---
    "tenure": {
        "display_name": "Customer tenure",
        "category": "engagement",
        "unit": "months",
        "aliases": ["customer_tenure", "months_as_customer", "account_age_months"],
        "direction_impact": {
            "high": "Long-running relationship usually means stickiness",
            "low": "Early relationship — still forming habit, easier to leave",
        },
        "action_hint_risk": "Invest in onboarding, early value moments, and a short check-in cadence.",
        "action_hint_protect": "Acknowledge tenure with recognition, not only discounts.",
    },
    "days_since_last_interaction": {
        "display_name": "Days since last contact",
        "category": "engagement",
        "unit": "days",
        "aliases": ["days_since_last_login", "recency", "last_contact_days"],
        "direction_impact": {
            "high": "Silence often signals fading habit",
            "low": "Recent activity shows the product is still in use",
        },
        "action_hint_risk": "Trigger a re-engagement touch with a concrete reason to return.",
        "action_hint_protect": "Keep useful moments coming without spam.",
    },
    "logins": {
        "display_name": "Login activity",
        "category": "engagement",
        "aliases": ["login_count", "app_usage", "sessions", "num_logins"],
        "direction_impact": {
            "high": "Heavy use usually means attachment",
            "low": "Light use is a classic early churn signal",
        },
        "action_hint_risk": "Surface unused value and remove activation friction.",
        "action_hint_protect": "Protect the experience that keeps them coming back.",
    },
    # --- money ---
    "monthly_charges": {
        "display_name": "Monthly charges",
        "category": "financial",
        "unit": "currency",
        "aliases": ["monthlycharges", "monthly_charge", "bill_amount", "mrr"],
        "direction_impact": {
            "high": "Higher bill can increase price sensitivity",
            "low": "Lower spend may mean thinner attachment or basic plan",
        },
        "action_hint_risk": "Review plan fit, remove unused paid extras, or offer a right-sized plan.",
        "action_hint_protect": "Reinforce value delivered for the fee — not only a discount.",
    },
    "total_charges": {
        "display_name": "Total charges",
        "category": "financial",
        "unit": "currency",
        "aliases": ["totalcharges", "lifetime_value", "ltv", "spend"],
        "direction_impact": {
            "high": "High lifetime spend is worth protecting",
            "low": "Limited lifetime spend — still building value",
        },
        "action_hint_risk": "Prioritize retention effort on high-lifetime relationships first.",
        "action_hint_protect": "Treat as high-value: white-glove save before aggressive discounting.",
    },
    "balance": {
        "display_name": "Account balance",
        "category": "financial",
        "aliases": ["account_balance", "wallet_balance"],
        "direction_impact": {
            "high": "Healthy balance often tracks commitment",
            "low": "Low balance may signal financial strain or wind-down",
        },
        "action_hint_risk": "Offer flexible payment options or right-sized plans.",
        "action_hint_protect": "Consider premium or loyalty extensions carefully.",
    },
    # --- commitment ---
    "contract": {
        "display_name": "Contract type",
        "category": "contract",
        "aliases": ["contract_type", "contracttype", "subscription_term", "plan_term"],
        "value_mapping": {
            "month-to-month": "Month-to-month can leave with little friction",
            "month_to_month": "Month-to-month can leave with little friction",
            "one year": "One-year term adds switching friction",
            "one_year": "One-year term adds switching friction",
            "two year": "Two-year term is usually the stickiest",
            "two_year": "Two-year term is usually the stickiest",
        },
        "action_hint_risk": "Offer a term upgrade for genuine value, not lock-in without benefit.",
        "action_hint_protect": "Honor loyalty without forcing a longer contract if unnecessary.",
    },
    "payment_method": {
        "display_name": "Payment method",
        "category": "contract",
        "aliases": ["paymentmethod", "pay_method"],
        "value_mapping": {
            "electronic check": "Manual e-check can mean more friction and churn",
            "electronic_check": "Manual e-check can mean more friction and churn",
            "mailed check": "Mailed check is high-friction billing",
            "mailed_check": "Mailed check is high-friction billing",
            "bank transfer (automatic)": "Auto-pay usually stabilizes the relationship",
            "credit card (automatic)": "Auto-pay usually stabilizes the relationship",
        },
        "action_hint_risk": "Move them to auto-pay and simplify billing surprises.",
        "action_hint_protect": "Keep billing transparent so auto-pay feels trustworthy.",
    },
    # --- product / services (telco-ish) ---
    "internet_service": {
        "display_name": "Internet service",
        "category": "services",
        "aliases": ["internetservice"],
        "value_mapping": {
            "fiber optic": "Premium connectivity — high value, higher expectations",
            "fiber_optic": "Premium connectivity — high value, higher expectations",
            "dsl": "Standard connectivity",
            "no": "No internet bundle — thinner product attachment",
        },
        "action_hint_risk": "Fix reliability and support for premium tiers first.",
        "action_hint_protect": "Defend premium experience; don't undercut with poor service quality.",
    },
    "online_security": {
        "display_name": "Online security add-on",
        "category": "services",
        "aliases": ["onlinesecurity"],
        "action_hint_risk": "Offer a low-friction security trial if it fills a real need.",
        "action_hint_protect": "Remind them of the protection already active.",
    },
    "tech_support": {
        "display_name": "Tech support",
        "category": "services",
        "aliases": ["techsupport"],
        "action_hint_risk": "Proactive support or a short diagnostic can reduce quiet frustration.",
        "action_hint_protect": "Keep support quality high — it is part of the stickiness.",
    },
    "online_backup": {
        "display_name": "Online backup",
        "category": "services",
        "aliases": ["onlinebackup"],
    },
    "device_protection": {
        "display_name": "Device protection",
        "category": "services",
        "aliases": ["deviceprotection"],
    },
    "streaming_tv": {
        "display_name": "Streaming TV",
        "category": "services",
        "aliases": ["streamingtv"],
    },
    "streaming_movies": {
        "display_name": "Streaming movies",
        "category": "services",
        "aliases": ["streamingmovies"],
    },
    "multiple_lines": {
        "display_name": "Multiple lines",
        "category": "services",
        "aliases": ["multiplelines"],
    },
    "paperless_billing": {
        "display_name": "Paperless billing",
        "category": "contract",
        "aliases": ["paperlessbilling"],
    },
    # --- support / health ---
    "num_support_tickets": {
        "display_name": "Support tickets",
        "category": "support",
        "aliases": ["tickets", "support_tickets", "numoftickets"],
        "direction_impact": {
            "high": "Ticket volume often tracks friction or unresolved issues",
            "low": "Few tickets can mean smooth use — or quiet disengagement",
        },
        "action_hint_risk": "Prioritize open loops and a human follow-up on recent tickets.",
        "action_hint_protect": "Ask for feedback while experience is still good.",
    },
    "num_complaints": {
        "display_name": "Complaints",
        "category": "support",
        "aliases": ["complaints"],
        "direction_impact": {
            "high": "Complaints are a direct dissatisfaction signal",
            "low": "Few complaints is a positive retention signal",
        },
        "action_hint_risk": "Escalate unresolved complaints with a single owner.",
        "action_hint_protect": "Capture testimonials while sentiment is favourable.",
    },
    "satisfaction_score": {
        "display_name": "Satisfaction score",
        "category": "support",
        "aliases": ["nps", "csat", "satisfaction"],
        "direction_impact": {
            "high": "High satisfaction anchors retention",
            "low": "Low satisfaction is a leading churn indicator",
        },
        "action_hint_risk": "Open a save conversation focused on the broken promise.",
        "action_hint_protect": "Invite advocacy or referral while scores are high.",
    },
    # --- banking-ish ---
    "is_active_member": {
        "display_name": "Active membership",
        "category": "engagement",
        "aliases": ["isactivemember", "active_member", "is_active"],
        "action_hint_risk": "Win-back sequence with a clear reason to re-activate.",
        "action_hint_protect": "Recognize active status with relevant rewards.",
    },
    "num_of_products": {
        "display_name": "Number of products",
        "category": "services",
        "aliases": ["numofproducts", "product_count", "products"],
        "direction_impact": {
            "high": "More products usually raise switching costs",
            "low": "Single-product ties are easier to sever",
        },
        "action_hint_risk": "Relevant cross-sell only if it solves a real job.",
        "action_hint_protect": "Deepen relationship with care, not product spam.",
    },
    "age": {
        "display_name": "Age",
        "category": "other",
        "aliases": ["customer_age"],
    },
    "credit_score": {
        "display_name": "Credit score",
        "category": "financial",
        "aliases": ["creditscore"],
    },
    "geography": {
        "display_name": "Geography",
        "category": "other",
        "aliases": ["region", "country", "state"],
    },
    "estimated_salary": {
        "display_name": "Estimated salary",
        "category": "financial",
        "aliases": ["estimatedsalary", "salary"],
    },
    "gender": {
        "display_name": "Gender",
        "category": "other",
        "aliases": [],
    },
    "partner": {
        "display_name": "Partner household",
        "category": "other",
        "aliases": ["has_partner"],
    },
    "dependents": {
        "display_name": "Dependents",
        "category": "other",
        "aliases": [],
    },
    "senior_citizen": {
        "display_name": "Senior citizen",
        "category": "other",
        "aliases": ["seniorcitizen"],
    },
    "phone_service": {
        "display_name": "Phone service",
        "category": "services",
        "aliases": ["phoneservice"],
    },
    # --- HR attrition (IBM Employee Attrition-style) ---
    "over_time": {
        "display_name": "Overtime",
        "category": "engagement",
        "aliases": ["overtime", "ot"],
        "value_mapping": {
            "yes": "Regular overtime often signals overload and burnout risk",
            "no": "No overtime — healthier hours baseline",
        },
        "action_hint_risk": "Cap overtime and rebalance workload with the manager.",
        "action_hint_protect": "Keep overtime exceptional, not the default.",
    },
    "job_satisfaction": {
        "display_name": "Job satisfaction",
        "category": "support",
        "aliases": ["jobsatisfaction"],
        "direction_impact": {
            "high": "Strong job satisfaction anchors retention",
            "low": "Low job satisfaction is a leading attrition signal",
        },
        "action_hint_risk": "Run a stay conversation focused on role fit and blockers.",
        "action_hint_protect": "Recognize what's working so it stays sticky.",
    },
    "environment_satisfaction": {
        "display_name": "Environment satisfaction",
        "category": "support",
        "aliases": ["environmentsatisfaction"],
        "direction_impact": {
            "high": "Healthy work environment supports staying",
            "low": "Poor environment accelerates exit intent",
        },
        "action_hint_risk": "Address team climate and manager support gaps quickly.",
        "action_hint_protect": "Protect culture habits that keep people engaged.",
    },
    "relationship_satisfaction": {
        "display_name": "Relationship satisfaction",
        "category": "support",
        "aliases": ["relationshipsatisfaction"],
        "action_hint_risk": "Coach the manager relationship or consider a team change.",
        "action_hint_protect": "Reinforce strong peer/manager bonds.",
    },
    "work_life_balance": {
        "display_name": "Work–life balance",
        "category": "engagement",
        "aliases": ["worklifebalance", "wlb"],
        "direction_impact": {
            "high": "Good balance reduces burnout-driven exits",
            "low": "Poor balance is a classic attrition driver",
        },
        "action_hint_risk": "Agree a concrete WLB plan (schedule, coverage, PTO).",
        "action_hint_protect": "Defend boundaries that keep balance healthy.",
    },
    "monthly_income": {
        "display_name": "Monthly income",
        "category": "financial",
        "unit": "currency",
        "aliases": ["monthlyincome", "compensation"],
        "direction_impact": {
            "high": "Higher pay can still leave if growth or hours are wrong",
            "low": "Below-market pay raises external offer risk",
        },
        "action_hint_risk": "Run a compensation market review against peers.",
        "action_hint_protect": "Keep pay transparent and aligned with contribution.",
    },
    "years_at_company": {
        "display_name": "Years at company",
        "category": "engagement",
        "aliases": ["yearsatcompany", "company_tenure"],
        "action_hint_risk": "Acknowledge tenure with growth, not only pay.",
        "action_hint_protect": "Celebrate loyalty with meaningful recognition.",
    },
    "years_since_last_promotion": {
        "display_name": "Years since last promotion",
        "category": "engagement",
        "aliases": ["yearssincelastpromotion"],
        "direction_impact": {
            "high": "Long promotion stall often drives external looking",
            "low": "Recent growth reduces restlessness",
        },
        "action_hint_risk": "Build a clear next-role / promotion path plan.",
        "action_hint_protect": "Keep the growth conversation visible.",
    },
    "years_in_current_role": {
        "display_name": "Years in current role",
        "category": "engagement",
        "aliases": ["yearsincurrentrole"],
        "action_hint_risk": "Explore internal mobility or role redesign.",
        "action_hint_protect": "Add stretch without forcing a hollow title change.",
    },
    "years_with_curr_manager": {
        "display_name": "Years with current manager",
        "category": "engagement",
        "aliases": ["yearswithcurrmanager"],
        "action_hint_risk": "Invest in the manager relationship or consider a move.",
        "action_hint_protect": "Protect a strong manager–IC pairing.",
    },
    "distance_from_home": {
        "display_name": "Distance from home",
        "category": "other",
        "aliases": ["distancefromhome", "commute_distance"],
        "direction_impact": {
            "high": "Long commute adds daily friction",
            "low": "Short commute is a quiet retention advantage",
        },
        "action_hint_risk": "Offer hybrid or remote flexibility where possible.",
        "action_hint_protect": "Keep commute-friendly arrangements stable.",
    },
    "business_travel": {
        "display_name": "Business travel",
        "category": "engagement",
        "aliases": ["businesstravel"],
        "value_mapping": {
            "travel_frequently": "Heavy travel load can erode balance and loyalty",
            "travel_rarely": "Light travel is usually manageable",
            "non-travel": "No travel requirement",
        },
        "action_hint_risk": "Reduce travel load or rotate coverage.",
        "action_hint_protect": "Keep travel predictable and compensated fairly.",
    },
    "job_involvement": {
        "display_name": "Job involvement",
        "category": "engagement",
        "aliases": ["jobinvolvement"],
        "action_hint_risk": "Increase ownership via mentoring or meaningful projects.",
        "action_hint_protect": "Keep high-involvement people challenged, not overloaded.",
    },
    "training_times_last_year": {
        "display_name": "Training last year",
        "category": "engagement",
        "aliases": ["trainingtimeslastyear", "training"],
        "action_hint_risk": "Fund learning & development to rebuild growth signal.",
        "action_hint_protect": "Maintain a visible development cadence.",
    },
    "percent_salary_hike": {
        "display_name": "Percent salary hike",
        "category": "financial",
        "aliases": ["percentsalaryhike", "salary_hike"],
        "action_hint_risk": "Revisit raise timing or market adjustment.",
        "action_hint_protect": "Keep hike communication clear and fair.",
    },
    "stock_option_level": {
        "display_name": "Stock option level",
        "category": "financial",
        "aliases": ["stockoptionlevel"],
        "action_hint_risk": "Discuss equity refresh if options are thin for the level.",
        "action_hint_protect": "Keep equity philosophy transparent.",
    },
    "job_level": {
        "display_name": "Job level",
        "category": "other",
        "aliases": ["joblevel"],
    },
}


def _build_alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for canon, info in FEATURE_MAPPING.items():
        index[_normalize_key(canon)] = canon
        for alias in info.get("aliases") or []:
            index[_normalize_key(alias)] = canon
    return index


_ALIAS_INDEX = _build_alias_index()


def resolve_feature_key(feature_name: str) -> Optional[str]:
    """Map arbitrary column name → canonical mapping key."""
    n = _normalize_key(feature_name)
    if n in _ALIAS_INDEX:
        return _ALIAS_INDEX[n]
    # strip common prefixes
    for prefix in ("feat_", "f_", "x_"):
        if n.startswith(prefix) and n[len(prefix) :] in _ALIAS_INDEX:
            return _ALIAS_INDEX[n[len(prefix) :]]
    # substring match as last resort (short keys only)
    for key in FEATURE_MAPPING:
        if key in n or n in key:
            return key
    return None


def get_feature_info(feature_name: str) -> dict[str, Any]:
    canon = resolve_feature_key(feature_name)
    if canon and canon in FEATURE_MAPPING:
        info = dict(FEATURE_MAPPING[canon])
        info["canonical"] = canon
        return info
    return {
        "display_name": _humanize(feature_name),
        "business_concept": _humanize(feature_name),
        "category": "other",
        "canonical": None,
    }


def _humanize(name: str) -> str:
    return _normalize_key(name).replace("_", " ").strip().title()


def get_value_interpretation(feature_name: str, value: Any) -> str:
    info = get_feature_info(feature_name)
    if value is None:
        return f"{info['display_name']}: unknown"

    if "value_mapping" in info:
        raw = str(value).strip().lower().replace(" ", "_")
        vm = info["value_mapping"]
        if raw in vm:
            return vm[raw]
        # try spaced form
        spaced = str(value).strip().lower()
        if spaced in vm:
            return vm[spaced]

    if "direction_impact" in info:
        try:
            num = float(value)
            thr = get_feature_thresholds(feature_name)
            if num >= thr["high"]:
                return info["direction_impact"]["high"]
            if num <= thr["low"]:
                return info["direction_impact"]["low"]
        except (TypeError, ValueError):
            pass

    return f"{info['display_name']}: {value}"


def get_feature_thresholds(feature_name: str) -> dict[str, float]:
    canon = resolve_feature_key(feature_name) or _normalize_key(feature_name)
    thresholds = {
        "tenure": {"low": 6, "high": 36},
        "monthly_charges": {"low": 30, "high": 80},
        "total_charges": {"low": 500, "high": 3000},
        "num_support_tickets": {"low": 1, "high": 5},
        "num_complaints": {"low": 0, "high": 2},
        "days_since_last_interaction": {"low": 7, "high": 60},
        "satisfaction_score": {"low": 2.5, "high": 4.0},
        "logins": {"low": 2, "high": 15},
        "num_of_products": {"low": 1, "high": 3},
    }
    return thresholds.get(canon, {"low": 0, "high": 100})


def get_feature_category(feature_name: str) -> str:
    return get_feature_info(feature_name).get("category", "other")


def get_action_hint(feature_name: str, raises_risk: bool) -> str:
    info = get_feature_info(feature_name)
    if raises_risk:
        return info.get(
            "action_hint_risk",
            f"Look for practical ways to improve {_humanize(feature_name).lower()}.",
        )
    return info.get(
        "action_hint_protect",
        f"Protect what is working on {_humanize(feature_name).lower()}.",
    )


def get_all_categories() -> list[str]:
    return sorted({info["category"] for info in FEATURE_MAPPING.values()})
