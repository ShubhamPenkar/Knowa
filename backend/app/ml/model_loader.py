"""Load trained artifacts regardless of routing strategy."""

from __future__ import annotations

import json
import os
from typing import Union

from app.ml.models.ensemble_model import EnsembleModel
from app.ml.models.foundation_model import FoundationModel
from app.config import get_settings

RoutedModel = Union[EnsembleModel, FoundationModel]


def detect_strategy(model_path: str) -> str:
    """Infer strategy from artifacts on disk (route_meta or meta jobs)."""
    route_path = os.path.join(model_path, "route_meta.json")
    if os.path.exists(route_path):
        try:
            with open(route_path, encoding="utf-8") as f:
                data = json.load(f)
            strategy = data.get("strategy")
            if strategy in ("foundation_model", "ensemble"):
                return strategy
        except Exception:
            pass

    if os.path.exists(os.path.join(model_path, "foundation_meta.joblib")):
        return "foundation_model"
    if os.path.exists(os.path.join(model_path, "ensemble_meta.joblib")):
        return "ensemble"
    # Legacy ensemble-only saves
    return "ensemble"


def write_route_meta(
    model_path: str,
    strategy: str,
    *,
    reason: str = "",
    extra: dict | None = None,
) -> None:
    os.makedirs(model_path, exist_ok=True)
    payload = {
        "strategy": strategy,
        "reason": reason,
        **(extra or {}),
    }
    with open(os.path.join(model_path, "route_meta.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _version_mismatch_hint(exc: BaseException) -> str:
    msg = str(exc)
    if "BitGenerator" in msg or "PCG64" in msg or "state must be a dict" in msg:
        return (
            f"{msg}. Saved model was pickled with a different NumPy/scikit-learn "
            "than this process (common when training with one venv and serving "
            "with another). Retrain the project using the same environment as "
            "the API server, or run the server from that training environment."
        )
    return msg


def load_routed_model(
    model_path: str,
    problem_type: str = "binary_classification",
) -> RoutedModel:
    """Instantiate and load the correct model type for a saved directory."""
    settings = get_settings()
    strategy = detect_strategy(model_path)

    try:
        if strategy == "foundation_model":
            model: RoutedModel = FoundationModel(
                problem_type=problem_type,
                conformal_alpha=settings.conformal_alpha,
                disagreement_threshold=settings.disagreement_threshold,
                interval_width_threshold=settings.interval_width_threshold,
                prefer_tabpfn=settings.prefer_tabpfn,
                tabpfn_max_rows=settings.foundation_max_rows,
                tabpfn_max_features=settings.foundation_max_features,
                probability_calibration=settings.probability_calibration,
            )
            model.load(model_path)
        else:
            model = EnsembleModel(
                problem_type=problem_type,
                n_folds=settings.stacking_n_folds,
                conformal_alpha=settings.conformal_alpha,
                disagreement_threshold=settings.disagreement_threshold,
                interval_width_threshold=settings.interval_width_threshold,
                enable_optuna=settings.enable_optuna,
                optuna_trials=settings.optuna_trials,
                optuna_timeout=settings.optuna_timeout_seconds,
                probability_calibration=settings.probability_calibration,
                early_stopping_rounds=settings.early_stopping_rounds,
            )
            model.load(model_path)
    except (ValueError, TypeError) as exc:
        raise ValueError(_version_mismatch_hint(exc)) from exc

    # Live policy wins over frozen train-time thresholds (quantile stays fitted)
    if hasattr(model, "calibrator") and model.calibrator is not None:
        model.calibrator.apply_policy(
            disagreement_threshold=settings.disagreement_threshold,
            interval_width_threshold=settings.interval_width_threshold,
        )
    return model


def build_model_for_strategy(
    strategy: str,
    problem_type: str = "binary_classification",
) -> RoutedModel:
    """Create an untrained model instance for the given routing strategy."""
    settings = get_settings()
    if strategy == "foundation_model":
        return FoundationModel(
            problem_type=problem_type,
            conformal_alpha=settings.conformal_alpha,
            disagreement_threshold=settings.disagreement_threshold,
            interval_width_threshold=settings.interval_width_threshold,
            prefer_tabpfn=settings.prefer_tabpfn,
            tabpfn_max_rows=settings.foundation_max_rows,
            tabpfn_max_features=settings.foundation_max_features,
            probability_calibration=settings.probability_calibration,
        )
    return EnsembleModel(
        problem_type=problem_type,
        n_folds=settings.stacking_n_folds,
        conformal_alpha=settings.conformal_alpha,
        disagreement_threshold=settings.disagreement_threshold,
        interval_width_threshold=settings.interval_width_threshold,
        enable_optuna=settings.enable_optuna,
        optuna_trials=settings.optuna_trials,
        optuna_timeout=settings.optuna_timeout_seconds,
        probability_calibration=settings.probability_calibration,
        early_stopping_rounds=settings.early_stopping_rounds,
    )
