"""Lightweight Optuna hyperparameter search on OOF/CV scores (Phase 1.5)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score


def _cv_scorer(problem_type: str) -> str:
    return "neg_root_mean_squared_error" if problem_type == "regression" else "roc_auc"


def tune_tree_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    problem_type: str = "binary_classification",
    family: str = "random_forest",
    n_trials: int = 12,
    timeout: Optional[float] = 60.0,
    random_state: int = 42,
    n_folds: int = 3,
) -> dict[str, Any]:
    """
    Optuna search for tree hyperparameters. Returns param dict for the family.

    Falls back to strong regularized defaults if Optuna is unavailable or
    the search fails.
    """
    defaults = _default_params(family, problem_type)
    if n_trials <= 0:
        return defaults

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        return defaults

    y_arr = np.asarray(y)
    if problem_type != "regression" and len(np.unique(y_arr)) < 2:
        return defaults

    # Subsample large frames for faster search
    X_use, y_use = X, y
    if len(X) > 4000:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=4000, replace=False)
        X_use = X.iloc[idx]
        y_use = y.iloc[idx] if hasattr(y, "iloc") else y_arr[idx]

    cv = (
        KFold(n_splits=min(n_folds, 3), shuffle=True, random_state=random_state)
        if problem_type == "regression"
        else StratifiedKFold(n_splits=min(n_folds, 3), shuffle=True, random_state=random_state)
    )
    scoring = _cv_scorer(problem_type)

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest_params(trial, family, problem_type, random_state)
        model = _build_sklearn_model(family, problem_type, params)
        try:
            scores = cross_val_score(
                model,
                X_use,
                y_use,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
            )
            return float(np.nanmean(scores))
        except Exception:
            return -1e9 if problem_type == "regression" else 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    if not study.best_trial or study.best_value is None or study.best_value <= -1e8:
        return defaults

    best = dict(study.best_params)
    # Optuna may omit fixed fields
    merged = {**defaults, **best}
    merged["_optuna_best_score"] = float(study.best_value)
    merged["_optuna_trials"] = len(study.trials)
    return merged


def _default_params(family: str, problem_type: str) -> dict[str, Any]:
    if family == "xgboost":
        return {
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "random_state": 42,
        }
    if family == "lightgbm":
        return {
            "num_leaves": 24,
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_samples": 30,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
        }
    # random_forest
    return {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_split": 8,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "random_state": 42,
    }


def _suggest_params(trial, family: str, problem_type: str, random_state: int) -> dict[str, Any]:
    if family == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "random_state": random_state,
        }
    if family == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 16, 48),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 60),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
            "random_state": random_state,
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 12),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        "random_state": random_state,
    }


def _build_sklearn_model(family: str, problem_type: str, params: dict[str, Any]):
    clean = {k: v for k, v in params.items() if not str(k).startswith("_")}
    if family == "xgboost":
        from app.ml.models.xgboost_model import _import_xgboost
        xgb = _import_xgboost()
        if problem_type == "regression":
            return xgb.XGBRegressor(**{**clean, "n_jobs": 1, "verbosity": 0})
        return xgb.XGBClassifier(
            **{**clean, "n_jobs": 1, "verbosity": 0, "use_label_encoder": False, "eval_metric": "logloss"}
        )
    if family == "lightgbm":
        from app.ml.models.lightgbm_model import _import_lightgbm
        lgb = _import_lightgbm()
        if problem_type == "regression":
            return lgb.LGBMRegressor(**{**clean, "n_jobs": 1, "verbose": -1})
        return lgb.LGBMClassifier(**{**clean, "n_jobs": 1, "verbose": -1})
    # RF
    if problem_type == "regression":
        return RandomForestRegressor(**{**clean, "n_jobs": 1})
    return RandomForestClassifier(**{**clean, "n_jobs": 1, "class_weight": "balanced"})
