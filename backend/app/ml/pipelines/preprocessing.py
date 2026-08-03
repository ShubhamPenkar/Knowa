"""Feature preprocessing and data preparation."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Feature configuration
NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "days_since_last_interaction",
    "num_complaints",
    "satisfaction_score",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "internet_service",
    "online_security",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

# Default encodings for categorical features
CATEGORICAL_ENCODINGS = {
    "contract_type": {"month-to-month": 0, "one_year": 1, "two_year": 2},
    "payment_method": {
        "electronic_check": 0,
        "mailed_check": 1,
        "bank_transfer": 2,
        "credit_card": 3,
    },
    "internet_service": {"no": 0, "dsl": 1, "fiber_optic": 2},
    "online_security": {"no_internet": 0, "no": 1, "yes": 2},
    "tech_support": {"no_internet": 0, "no": 1, "yes": 2},
    "streaming_tv": {"no_internet": 0, "no": 1, "yes": 2},
    "streaming_movies": {"no_internet": 0, "no": 1, "yes": 2},
}


def preprocess_features(
    features: dict[str, Any],
    scaler: Optional[StandardScaler] = None
) -> pd.DataFrame:
    """
    Preprocess a single feature dictionary for prediction.
    
    Args:
        features: Feature dictionary
        scaler: Optional fitted scaler for numeric features
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            encoding = CATEGORICAL_ENCODINGS.get(col, {})
            # Handle unknown values
            df[col] = df[col].map(lambda x: encoding.get(str(x).lower().replace(" ", "_"), 0))
    
    # Ensure numeric types
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Ensure all expected columns exist
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for col in all_features:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns
    df = df[all_features]
    
    return df


def preprocess_dataframe(
    df: pd.DataFrame,
    target_column: str = "churn",
    fit_scaler: bool = False
) -> tuple[pd.DataFrame, pd.Series, Optional[StandardScaler]]:
    """
    Preprocess DataFrame for training.
    
    Args:
        df: Raw DataFrame
        target_column: Name of target column
        fit_scaler: Whether to fit and return a scaler
        
    Returns:
        Tuple of (X, y, scaler)
    """
    df = df.copy()
    
    # Extract target
    if target_column in df.columns:
        y = df[target_column].astype(int)
        df = df.drop(columns=[target_column])
    else:
        y = pd.Series([0] * len(df))
    
    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            encoding = CATEGORICAL_ENCODINGS.get(col, {})
            df[col] = df[col].fillna("unknown").astype(str).str.lower().str.replace(" ", "_")
            df[col] = df[col].map(lambda x: encoding.get(x, 0))
    
    # Handle numeric features
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Ensure all columns exist
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for col in all_features:
        if col not in df.columns:
            df[col] = 0
    
    X = df[all_features]
    
    # Optionally fit scaler
    scaler = None
    if fit_scaler:
        scaler = StandardScaler()
        X[NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])
    
    return X, y, scaler


def generate_sample_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic churn dataset for testing.
    
    Creates realistic-looking telecom churn data.
    """
    np.random.seed(random_state)
    
    data = {
        "tenure": np.random.exponential(scale=24, size=n_samples).clip(1, 72).astype(int),
        "monthly_charges": np.random.normal(65, 25, n_samples).clip(20, 120),
        "contract_type": np.random.choice(
            ["month-to-month", "one_year", "two_year"],
            size=n_samples,
            p=[0.5, 0.3, 0.2]
        ),
        "payment_method": np.random.choice(
            ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
            size=n_samples,
            p=[0.35, 0.15, 0.25, 0.25]
        ),
        "internet_service": np.random.choice(
            ["fiber_optic", "dsl", "no"],
            size=n_samples,
            p=[0.45, 0.35, 0.2]
        ),
        "online_security": np.random.choice(
            ["yes", "no", "no_internet"],
            size=n_samples,
            p=[0.3, 0.5, 0.2]
        ),
        "tech_support": np.random.choice(
            ["yes", "no", "no_internet"],
            size=n_samples,
            p=[0.3, 0.5, 0.2]
        ),
        "streaming_tv": np.random.choice(
            ["yes", "no", "no_internet"],
            size=n_samples,
            p=[0.4, 0.4, 0.2]
        ),
        "streaming_movies": np.random.choice(
            ["yes", "no", "no_internet"],
            size=n_samples,
            p=[0.4, 0.4, 0.2]
        ),
        "num_support_tickets": np.random.poisson(2, n_samples),
        "num_complaints": np.random.poisson(0.5, n_samples),
        "days_since_last_interaction": np.random.exponential(30, n_samples).clip(0, 180).astype(int),
        "satisfaction_score": np.random.normal(3.5, 0.8, n_samples).clip(1, 5),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate total charges
    df["total_charges"] = df["tenure"] * df["monthly_charges"]
    
    # Generate churn labels based on features (realistic correlations)
    churn_prob = np.zeros(n_samples)
    
    # Contract type effect
    churn_prob += (df["contract_type"] == "month-to-month").astype(float) * 0.3
    churn_prob -= (df["contract_type"] == "two_year").astype(float) * 0.2
    
    # Tenure effect (lower tenure = higher churn)
    churn_prob += (df["tenure"] < 12).astype(float) * 0.2
    churn_prob -= (df["tenure"] > 36).astype(float) * 0.15
    
    # Payment method effect
    churn_prob += (df["payment_method"] == "electronic_check").astype(float) * 0.15
    
    # Service effects
    churn_prob -= (df["tech_support"] == "yes").astype(float) * 0.1
    churn_prob -= (df["online_security"] == "yes").astype(float) * 0.1
    
    # Satisfaction effect
    churn_prob -= (df["satisfaction_score"] > 4).astype(float) * 0.2
    churn_prob += (df["satisfaction_score"] < 3).astype(float) * 0.25
    
    # Complaints effect
    churn_prob += (df["num_complaints"] > 0).astype(float) * 0.15
    churn_prob += (df["num_complaints"] > 2).astype(float) * 0.15
    
    # Monthly charges effect
    churn_prob += (df["monthly_charges"] > 80).astype(float) * 0.1
    
    # Normalize to 0-1 and add noise
    churn_prob = (churn_prob - churn_prob.min()) / (churn_prob.max() - churn_prob.min() + 0.001)
    churn_prob = churn_prob * 0.7 + 0.15  # Scale to 15-85% range
    churn_prob += np.random.normal(0, 0.1, n_samples)
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Generate binary labels
    df["churn"] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    return df


def get_feature_names() -> list[str]:
    """Get all feature names in order."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES
