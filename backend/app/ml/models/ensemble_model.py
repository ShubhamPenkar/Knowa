"""Ensemble model with stacking meta-learner (Phase 1a/1.5).

Base models: XGBoost, LightGBM, Random Forest, Logistic/Linear.
Combination: stacking meta-learner trained on out-of-fold base predictions.

Phase 1.5:
  - Optional Optuna HPO on tree bases
  - Early stopping with holdout during refit
  - Isotonic/Platt probability calibration
  - Conformal residual intervals + disagreement abstention
"""

from __future__ import annotations

import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from app.ml.calibration import ConformalCalibrator, UncertaintyResult
from app.ml.models.base_model import BaseModel
from app.ml.models.lightgbm_model import LightGBMModel
from app.ml.models.logistic_model import LogisticModel
from app.ml.models.random_forest_model import RandomForestModel
from app.ml.models.xgboost_model import XGBoostModel
from app.ml.probability_calibration import ProbabilityCalibrator
from app.ml.tuning import tune_tree_hyperparameters


class EnsembleModel(BaseModel):
    """Stacked ensemble with probability + conformal calibration."""

    BASE_NAMES = ("xgboost", "lightgbm", "random_forest", "logistic")

    def __init__(
        self,
        version: str = "1.0",
        problem_type: str = "binary_classification",
        weights: Optional[dict[str, float]] = None,
        n_folds: int = 5,
        conformal_alpha: float = 0.1,
        disagreement_threshold: float = 0.25,
        interval_width_threshold: float = 0.85,
        random_state: int = 42,
        enable_optuna: bool = True,
        optuna_trials: int = 12,
        optuna_timeout: float = 60.0,
        probability_calibration: str = "isotonic",
        early_stopping_rounds: int = 40,
    ):
        super().__init__("ensemble", version)
        self.problem_type = problem_type
        self.n_folds = n_folds
        self.random_state = random_state
        self.enable_optuna = enable_optuna
        self.optuna_trials = optuna_trials
        self.optuna_timeout = optuna_timeout
        self.probability_calibration_method = probability_calibration
        self.early_stopping_rounds = early_stopping_rounds
        self.tuned_params: dict[str, dict] = {}

        self.weights = weights or {
            "xgboost": 0.35,
            "lightgbm": 0.35,
            "random_forest": 0.20,
            "logistic": 0.10,
        }

        self.models = self._new_base_models()
        self.meta_learner = None
        self.use_stacking = True
        self.model_names = list(self.models.keys())
        self.calibrator = ConformalCalibrator(
            alpha=conformal_alpha,
            problem_type=problem_type,
            disagreement_threshold=disagreement_threshold,
            interval_width_threshold=interval_width_threshold,
        )
        self.prob_calibrator = ProbabilityCalibrator(method=probability_calibration)  # type: ignore[arg-type]

    def _is_regression(self) -> bool:
        return self.problem_type == "regression"

    def _new_base_models(self, param_overrides: Optional[dict[str, dict]] = None) -> dict[str, BaseModel]:
        param_overrides = param_overrides if param_overrides is not None else getattr(self, "tuned_params", {}) or {}
        builders = {
            "xgboost": XGBoostModel,
            "lightgbm": LightGBMModel,
            "random_forest": RandomForestModel,
            "logistic": LogisticModel,
        }
        models: dict[str, BaseModel] = {}
        for name, cls in builders.items():
            try:
                kwargs = {k: v for k, v in dict(param_overrides.get(name, {})).items() if not str(k).startswith("_")}
                if name in ("xgboost", "lightgbm"):
                    kwargs.setdefault("early_stopping_rounds", self.early_stopping_rounds)
                models[name] = cls(version=self.version, problem_type=self.problem_type, **kwargs)
            except Exception as e:
                print(f"Skipping base model '{name}': {e}")
        if not models:
            raise RuntimeError(
                "No base models available. Install scikit-learn, and ideally "
                "xgboost/lightgbm (macOS: brew install libomp)."
            )
        self.model_names = list(models.keys())
        available_w = {k: self.weights.get(k, 1.0) for k in models}
        total = sum(available_w.values()) or 1.0
        self.weights = {k: v / total for k, v in available_w.items()}
        return models

    def _tune_bases(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not self.enable_optuna or self.optuna_trials <= 0:
            return
        probe = self._new_base_models({})
        for family in ("xgboost", "lightgbm", "random_forest"):
            if family not in probe:
                continue
            try:
                print(f"Optuna tuning {family} ({self.optuna_trials} trials)...")
                best = tune_tree_hyperparameters(
                    X,
                    y,
                    problem_type=self.problem_type,
                    family=family,
                    n_trials=self.optuna_trials,
                    timeout=self.optuna_timeout,
                    random_state=self.random_state,
                )
                self.tuned_params[family] = best
                print(f"  {family} best CV score={best.get('_optuna_best_score')}")
            except Exception as e:
                print(f"  Optuna skip {family}: {e}")

    def _base_outputs(self, models: dict[str, BaseModel], X: pd.DataFrame) -> np.ndarray:
        cols = []
        for name in self.model_names:
            if name not in models:
                continue
            model = models[name]
            if self._is_regression():
                cols.append(model.predict(X).astype(float))
            else:
                cols.append(model.predict_proba(X).astype(float))
        return np.column_stack(cols)

    def _disagreement(self, X: pd.DataFrame) -> np.ndarray:
        outputs = self._base_outputs(self.models, X)
        return np.std(outputs, axis=1)

    def _build_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        return self._base_outputs(self.models, X)

    def _meta_predict(self, meta_X: np.ndarray) -> np.ndarray:
        if self.meta_learner is None or not self.use_stacking:
            w = np.array([self.weights.get(n, 0.0) for n in self.model_names], dtype=float)
            w = w / (w.sum() + 1e-12)
            return meta_X @ w
        if self._is_regression():
            return self.meta_learner.predict(meta_X).astype(float)
        return self.meta_learner.predict_proba(meta_X)[:, 1].astype(float)

    def _raw_stacked(self, X: pd.DataFrame) -> np.ndarray:
        return self._meta_predict(self._build_meta_features(X))

    def _generate_oof_predictions(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        n = len(X)
        n_models = len(self.model_names)
        oof = np.zeros((n, n_models), dtype=float)

        n_folds = min(self.n_folds, max(2, n // 10)) if n >= 20 else 2
        n_folds = min(n_folds, n)

        y_arr = np.asarray(y)
        if self._is_regression():
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            splits = splitter.split(X)
        else:
            try:
                splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
                splits = splitter.split(X, y_arr)
            except ValueError:
                splitter = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
                splits = splitter.split(X)

        X_reset = X.reset_index(drop=True)
        y_reset = pd.Series(y_arr).reset_index(drop=True)

        for train_idx, val_idx in splits:
            X_tr, X_va = X_reset.iloc[train_idx], X_reset.iloc[val_idx]
            y_tr = y_reset.iloc[train_idx]
            fold_models = self._new_base_models(self.tuned_params)
            for name, model in fold_models.items():
                model.train(X_tr, y_tr)
            oof[val_idx] = self._base_outputs(fold_models, X_va)

        return oof

    def _fit_meta_learner(self, oof: np.ndarray, y: pd.Series) -> None:
        y_arr = np.asarray(y).ravel()
        if self._is_regression():
            meta = Ridge(alpha=2.0)
            meta.fit(oof, y_arr)
            self.meta_learner = meta
            self.use_stacking = True
            return

        if len(np.unique(y_arr)) < 2:
            self.use_stacking = False
            self.meta_learner = None
            return

        meta = LogisticRegression(
            max_iter=2000,
            C=0.5,
            solver="lbfgs",
            random_state=self.random_state,
        )
        meta.fit(oof, y_arr.astype(int))
        self.meta_learner = meta
        self.use_stacking = True

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
        calibration_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> dict[str, float]:
        self.feature_names = list(X.columns)
        all_metrics: dict[str, float] = {}

        self._tune_bases(X, y)
        self.models = self._new_base_models(self.tuned_params)
        oof = self._generate_oof_predictions(X, y)
        self._fit_meta_learner(oof, y)

        self.models = self._new_base_models(self.tuned_params)
        X_fit, y_fit = X, y
        es_holdout = None
        if len(X) >= 40:
            try:
                strat = y if not self._is_regression() else None
                X_fit, X_es, y_fit, y_es = train_test_split(
                    X, y, test_size=0.15, random_state=self.random_state, stratify=strat
                )
                if self._is_regression() or len(np.unique(y_fit)) > 1:
                    es_holdout = (X_es, y_es)
                else:
                    X_fit, y_fit, es_holdout = X, y, None
            except ValueError:
                X_fit, y_fit, es_holdout = X, y, None

        for name, model in self.models.items():
            metrics = model.train(X_fit, y_fit, validation_data=es_holdout)
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.floating)):
                    all_metrics[f"{name}_{key}"] = float(value)

        self.is_trained = True

        if calibration_data is not None:
            X_cal, y_cal = calibration_data
            raw_cal = self._raw_stacked(X_cal)
            cal_true = np.asarray(y_cal, dtype=float)
        elif validation_data is not None:
            X_cal, y_cal = validation_data
            raw_cal = self._raw_stacked(X_cal)
            cal_true = np.asarray(y_cal, dtype=float)
        else:
            raw_cal = self._meta_predict(oof)
            cal_true = np.asarray(y, dtype=float)

        if not self._is_regression():
            self.prob_calibrator.fit(cal_true, raw_cal)
            cal_pred = self.prob_calibrator.transform(raw_cal)
        else:
            self.prob_calibrator = ProbabilityCalibrator(method="none")
            cal_pred = raw_cal

        try:
            self.calibrator.fit(cal_true, cal_pred)
        except ValueError:
            self.calibrator.fit(
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]) if self._is_regression() else np.array([0.05, 0.95]),
            )

        all_metrics["stacking_enabled"] = float(self.use_stacking)
        all_metrics["conformal_quantile"] = float(self.calibrator.quantile or 0.0)
        all_metrics["conformal_coverage_target"] = float(self.calibrator.coverage_level)
        all_metrics["prob_calibration_fitted"] = float(self.prob_calibrator.is_fitted)
        all_metrics["optuna_ran"] = float(bool(self.tuned_params))

        if self._is_regression():
            y_pred = self.predict(X)
            self.training_metrics = {
                "ensemble_mae": mean_absolute_error(y, y_pred),
                "ensemble_mse": mean_squared_error(y, y_pred),
                "ensemble_rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "ensemble_r2_score": r2_score(y, y_pred),
                **all_metrics,
            }
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                self.training_metrics.update({
                    "ensemble_test_mae": mean_absolute_error(y_val, y_val_pred),
                    "ensemble_test_rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
                    "ensemble_test_r2_score": r2_score(y_val, y_val_pred),
                    "ensemble_mae": mean_absolute_error(y_val, y_val_pred),
                    "ensemble_mse": mean_squared_error(y_val, y_val_pred),
                    "ensemble_rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
                    "ensemble_r2_score": r2_score(y_val, y_val_pred),
                })
        else:
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)
            self.training_metrics = {
                "ensemble_accuracy": accuracy_score(y, y_pred),
                "ensemble_precision": precision_score(y, y_pred, zero_division=0),
                "ensemble_recall": recall_score(y, y_pred, zero_division=0),
                "ensemble_f1_score": f1_score(y, y_pred, zero_division=0),
                "ensemble_auc_roc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
                "ensemble_brier": float(brier_score_loss(y, y_proba)),
                **all_metrics,
            }
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                y_val_proba = self.predict_proba(X_val)
                auc = roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.5
                self.training_metrics.update({
                    "ensemble_test_accuracy": accuracy_score(y_val, y_val_pred),
                    "ensemble_test_precision": precision_score(y_val, y_val_pred, zero_division=0),
                    "ensemble_test_recall": recall_score(y_val, y_val_pred, zero_division=0),
                    "ensemble_test_f1_score": f1_score(y_val, y_val_pred, zero_division=0),
                    "ensemble_test_auc_roc": auc,
                    "ensemble_test_brier": float(brier_score_loss(y_val, y_val_proba)),
                    "ensemble_accuracy": accuracy_score(y_val, y_val_pred),
                    "ensemble_precision": precision_score(y_val, y_val_pred, zero_division=0),
                    "ensemble_recall": recall_score(y_val, y_val_pred, zero_division=0),
                    "ensemble_f1_score": f1_score(y_val, y_val_pred, zero_division=0),
                    "ensemble_auc_roc": auc,
                    "ensemble_brier": float(brier_score_loss(y_val, y_val_proba)),
                    "ensemble_val_accuracy": accuracy_score(y_val, y_val_pred),
                    "ensemble_val_auc_roc": auc,
                })

        if len(cal_pred):
            lower, upper, _ = self.calibrator.predict_interval(cal_pred)
            y_t = cal_true.ravel()
            covered = float(np.mean((y_t >= lower) & (y_t <= upper)))
            self.training_metrics["conformal_empirical_coverage_calib"] = covered
            self.training_metrics["conformal_mean_width_calib"] = float(np.mean(upper - lower))

        return self.training_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        if self._is_regression():
            return self.predict_proba(X)
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        raw = self._raw_stacked(X)
        if self._is_regression():
            return raw
        if getattr(self, "prob_calibrator", None) is not None:
            return self.prob_calibrator.transform(raw)
        return raw

    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        std = self._disagreement(X)
        if self._is_regression():
            preds = self._base_outputs(self.models, X)
            mean_pred = np.mean(preds, axis=1)
            cv = np.where(np.abs(mean_pred) > 1e-8, std / np.abs(mean_pred), std)
            return 1.0 / (1.0 + cv)
        return np.clip(1.0 - (std / 0.5), 0.0, 1.0)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> list[UncertaintyResult]:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        point = self.predict_proba(X)
        disagreement = self._disagreement(X)
        return [
            self.calibrator.evaluate_uncertainty(
                y_pred=float(point[i]),
                disagreement=float(disagreement[i]),
            )
            for i in range(len(X))
        ]

    def get_model_predictions(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        if self._is_regression():
            return {name: model.predict(X) for name, model in self.models.items()}
        return {name: model.predict_proba(X) for name, model in self.models.items()}

    def get_feature_importance(self) -> dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        importances: dict[str, float] = {}
        for name, model in self.models.items():
            weight = self.weights.get(name, 0)
            for feature, imp in model.get_feature_importance().items():
                importances[feature] = importances.get(feature, 0.0) + weight * imp
        return importances

    def get_primary_model(self) -> BaseModel:
        primary = max(self.weights.items(), key=lambda x: x[1])
        return self.models[primary[0]]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            model.save(os.path.join(path, f"{name}.joblib"))
        if self.meta_learner is not None:
            joblib.dump(self.meta_learner, os.path.join(path, "meta_learner.joblib"))
        self.calibrator.save(os.path.join(path, "calibrator.joblib"))
        if self.prob_calibrator is not None:
            self.prob_calibrator.save(os.path.join(path, "prob_calibrator.joblib"))
        metadata = {
            "weights": self.weights,
            "version": self.version,
            "problem_type": self.problem_type,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "use_stacking": self.use_stacking,
            "model_names": self.model_names,
            "n_folds": self.n_folds,
            "combination_method": "stacking" if self.use_stacking else "weighted_average",
            "tuned_params": self.tuned_params,
            "probability_calibration_method": self.probability_calibration_method,
        }
        joblib.dump(metadata, os.path.join(path, "ensemble_meta.joblib"))
        with open(os.path.join(path, "route_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "strategy": "ensemble",
                    "combination_method": metadata["combination_method"],
                    "problem_type": self.problem_type,
                    "version": self.version,
                    "model_names": self.model_names,
                },
                f,
                indent=2,
            )

    def load(self, path: str) -> None:
        metadata = joblib.load(os.path.join(path, "ensemble_meta.joblib"))
        self.weights = metadata["weights"]
        self.version = metadata["version"]
        self.problem_type = metadata.get("problem_type", "binary_classification")
        self.feature_names = metadata["feature_names"]
        self.training_metrics = metadata.get("training_metrics", {})
        self.use_stacking = metadata.get("use_stacking", False)
        self.model_names = metadata.get("model_names", list(self.BASE_NAMES))
        self.n_folds = metadata.get("n_folds", 5)
        self.tuned_params = metadata.get("tuned_params", {})
        self.probability_calibration_method = metadata.get("probability_calibration_method", "isotonic")

        loaded_models = {}
        for name in ("xgboost", "lightgbm", "random_forest", "logistic"):
            model_file = os.path.join(path, f"{name}.joblib")
            if not os.path.exists(model_file):
                continue
            cls = {
                "xgboost": XGBoostModel,
                "lightgbm": LightGBMModel,
                "random_forest": RandomForestModel,
                "logistic": LogisticModel,
            }[name]
            try:
                model = cls(version=self.version, problem_type=self.problem_type)
                model.load(model_file)
                loaded_models[name] = model
            except Exception as e:
                print(f"Could not load base model '{name}': {e}")
        if not loaded_models:
            raise RuntimeError(f"No ensemble component models found under {path}")
        self.models = loaded_models
        self.model_names = list(loaded_models.keys())

        meta_path = os.path.join(path, "meta_learner.joblib")
        if os.path.exists(meta_path):
            self.meta_learner = joblib.load(meta_path)
            self.use_stacking = True
        else:
            self.meta_learner = None
            self.use_stacking = False

        self.calibrator = ConformalCalibrator(problem_type=self.problem_type)
        cal_path = os.path.join(path, "calibrator.joblib")
        if os.path.exists(cal_path):
            self.calibrator.load(cal_path)
        else:
            self.calibrator.is_fitted = False

        self.prob_calibrator = ProbabilityCalibrator(method=self.probability_calibration_method)  # type: ignore[arg-type]
        pc_path = os.path.join(path, "prob_calibrator.joblib")
        if os.path.exists(pc_path):
            self.prob_calibrator.load(pc_path)

        self.is_trained = True
