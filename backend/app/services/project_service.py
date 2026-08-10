"""Project management service for SaaS."""

import os
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import (
    Project,
    Dataset,
    TrainedModel,
    CustomAction,
    ProjectPrediction,
    Decision,
)

# Short-lived cache for action-effectiveness blends on the predict hot path.
# Keyed by org:project; invalidated on feedback writes or when outcome count changes.
_EFFECTIVENESS_CACHE: dict[str, tuple[float, int, dict]] = {}
_EFFECTIVENESS_TTL_SEC = 60.0
from app.services.dataset_service import DatasetService
from app.ml.dataset_profiler import (
    ProfilingError,
    apply_feature_exclusions,
    profile_dataframe,
    suggest_positive_label,
)
from app.ml.threshold_tuning import (
    get_decision_threshold,
    set_decision_threshold_meta,
    tune_decision_threshold,
)

settings = get_settings()


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars so SQLAlchemy JSON columns can serialize."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if value is not None and not isinstance(value, (str, bytes, dict, list, tuple)) and pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    return value


class ProjectService:
    """Handle project creation, configuration, and training."""
    
    def __init__(self, db: Session, org_id: str):
        self.db = db
        self.org_id = org_id
        self.model_dir = os.path.join(settings.model_path, "projects", org_id)
        os.makedirs(self.model_dir, exist_ok=True)
    
    # =========================================================================
    # Project CRUD
    # =========================================================================
    
    def create_project(
        self,
        name: str,
        dataset_id: str,
        target_column: str,
        feature_columns: list[str],
        target_positive_label: str = "1",
        target_description: str = "outcome",
        problem_type: str = "binary_classification",
        description: Optional[str] = None,
    ) -> Project:
        """Create new prediction project (runs dataset profiler before save)."""
        dataset = self.db.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.organization_id == self.org_id,
        ).first()
        if not dataset:
            raise ValueError("Dataset not found")

        column_names = [col["name"] for col in dataset.columns]
        if target_column not in column_names:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        invalid_features = [f for f in feature_columns if f not in column_names]
        if invalid_features:
            raise ValueError(f"Feature columns not found: {invalid_features}")

        if target_column in feature_columns:
            raise ValueError("Target column cannot be in feature columns")

        dataset_service = DatasetService(self.db, self.org_id)
        df = dataset_service.load_dataframe(dataset_id)

        # Soft-adopt stale API default "1" only when a clear preferred label exists
        if problem_type != "regression":
            present = df[target_column].dropna().astype(str).unique().tolist()
            pos = str(target_positive_label)
            if pos not in present:
                suggested = suggest_positive_label(present)
                if pos == "1" and suggested is not None:
                    target_positive_label = suggested

        report = profile_dataframe(
            df,
            target_column=target_column,
            target_positive_label=(
                None if problem_type == "regression" else str(target_positive_label)
            ),
            problem_type=problem_type,
            feature_columns=list(feature_columns),
        )
        feature_columns = apply_feature_exclusions(list(feature_columns), report)
        if not feature_columns:
            raise ProfilingError(
                "All selected feature columns were excluded as identifiers or constants. "
                "Choose predictive features instead.",
                code="no_usable_features",
                blocking_issues=[
                    {
                        "code": "no_usable_features",
                        "excluded_as_id": report.excluded_as_id,
                        "dropped_as_constant": report.dropped_as_constant,
                    }
                ],
                warnings=report.warnings,
            )
        # Re-profile with cleaned features so ID exclusions don't false-block
        report = profile_dataframe(
            df,
            target_column=target_column,
            target_positive_label=(
                None if problem_type == "regression" else str(target_positive_label)
            ),
            problem_type=problem_type,
            feature_columns=list(feature_columns),
        )
        report.raise_if_blocking()

        feature_config = None
        if description:
            # Light B1 breadcrumb (preserved across train when feature_config is rebuilt)
            feature_config = {
                "intent": {
                    "source": "onboarding",
                    "target_description": target_description,
                    "problem_statement": str(description)[:500],
                }
            }

        project = Project(
            organization_id=self.org_id,
            dataset_id=dataset_id,
            name=name,
            description=description,
            target_column=target_column,
            target_positive_label=target_positive_label,
            target_description=target_description,
            problem_type=problem_type,
            feature_columns=feature_columns,
            feature_config=feature_config,
            status="draft",
        )
        self.db.add(project)
        self.db.commit()
        self.db.refresh(project)
        return project
    
    def list_projects(self) -> list[Project]:
        """List all projects for organization."""
        return self.db.query(Project).filter(
            Project.organization_id == self.org_id,
            Project.is_active == True
        ).order_by(Project.created_at.desc()).all()

    def org_health(self, project_id: Optional[str] = None) -> dict[str, Any]:
        """
        Lightweight org pulse for the homepage strip:
        follow-ups due, soft/low-trust cases, ready-project guidance quality.

        soft_cases is scoped to the focus project (requested or primary ready)
        so the strip matches Cases → Don't act → Logged for that project.
        soft_cases_org is the org-wide total.
        """
        from app.services.decision_service import DecisionService

        portfolio = DecisionService(self.db, self.org_id).list_portfolio(limit=1)
        counts = portfolio.get("counts") or {}
        overdue = int(counts.get("overdue") or 0)
        due_now = int(counts.get("due_now") or 0)

        ready = (
            self.db.query(Project)
            .filter(
                Project.organization_id == self.org_id,
                Project.is_active == True,
                Project.status.in_(["ready", "trained"]),
            )
            .all()
        )
        ready_ids = [p.id for p in ready]

        soft_cases_org = 0
        if ready_ids:
            soft_cases_org = (
                self.db.query(ProjectPrediction)
                .filter(
                    ProjectPrediction.project_id.in_(ready_ids),
                    ProjectPrediction.low_confidence.is_(True),
                )
                .count()
            )

        focus_id = None
        if project_id and project_id in ready_ids:
            focus_id = project_id
        elif ready_ids:
            focus_id = ready_ids[0]

        soft_cases = 0
        if focus_id:
            soft_cases = (
                self.db.query(ProjectPrediction)
                .filter(
                    ProjectPrediction.project_id == focus_id,
                    ProjectPrediction.low_confidence.is_(True),
                )
                .count()
            )

        # Rough model health from latest trained models
        rough = 0
        solid = 0
        for p in ready:
            tm = (
                self.db.query(TrainedModel)
                .filter(
                    TrainedModel.project_id == p.id,
                    TrainedModel.is_active == True,
                )
                .order_by(TrainedModel.trained_at.desc())
                .first()
            )
            if not tm:
                continue
            if p.problem_type == "regression":
                r2 = tm.r2_score
                if r2 is not None and float(r2) < 0.2:
                    rough += 1
                elif r2 is not None and float(r2) >= 0.5:
                    solid += 1
            else:
                score = tm.auc_roc if tm.auc_roc is not None else tm.accuracy
                if score is not None and float(score) < 0.6:
                    rough += 1
                elif score is not None and float(score) >= 0.7:
                    solid += 1

        due_attention = overdue + due_now
        bits = []
        if due_attention:
            bits.append(f"{due_attention} follow-up(s) need check-in")
        if soft_cases_org:
            bits.append(
                f"{soft_cases_org} don't-act case(s) logged — review before big spends"
            )
        if rough:
            bits.append(f"{rough} project(s) look like a rough guide")
        if not bits:
            bits.append(
                f"{len(ready)} ready project(s)"
                + (f", {solid} solid+" if solid else "")
                + ". Stack looks calm."
            )

        return {
            "layer": "org_health",
            "counts": {
                "overdue": overdue,
                "due_now": due_now,
                "due_attention": due_attention,
                "soft_cases": soft_cases,
                "soft_cases_org": soft_cases_org,
                "ready_projects": len(ready),
                "rough_models": rough,
                "solid_models": solid,
            },
            "plain_summary": " · ".join(bits),
            "primary_project_id": focus_id,
        }
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        return self.db.query(Project).filter(
            Project.id == project_id,
            Project.organization_id == self.org_id
        ).first()
    
    def update_project(self, project_id: str, updates: dict[str, Any]) -> Project:
        """Update project configuration."""
        project = self.get_project(project_id)
        if not project:
            raise ValueError("Project not found")
        
        allowed_fields = ["name", "description", "target_positive_label", "target_description", 
                          "feature_columns", "feature_config", "model_type"]
        
        # If feature list updated, strip ID/constant via profiler
        if "feature_columns" in updates and updates["feature_columns"] is not None:
            dataset_service = DatasetService(self.db, self.org_id)
            df = dataset_service.load_dataframe(project.dataset_id)
            feats = list(updates["feature_columns"])
            report = profile_dataframe(
                df,
                target_column=project.target_column,
                target_positive_label=(
                    updates.get("target_positive_label", project.target_positive_label)
                    if project.problem_type != "regression"
                    else None
                ),
                problem_type=project.problem_type,
                feature_columns=feats,
            )
            cleaned = apply_feature_exclusions(feats, report)
            if not cleaned:
                raise ProfilingError(
                    "All selected feature columns were excluded as identifiers or constants. "
                    "Choose predictive features instead.",
                    code="no_usable_features",
                    blocking_issues=[{"code": "no_usable_features"}],
                    warnings=report.warnings,
                )
            updates = {**updates, "feature_columns": cleaned}

        # If positive label updated, must appear in data (profiler hard-block)
        if "target_positive_label" in updates and updates["target_positive_label"] is not None:
            if project.problem_type != "regression":
                dataset_service = DatasetService(self.db, self.org_id)
                df = dataset_service.load_dataframe(project.dataset_id)
                report = profile_dataframe(
                    df,
                    target_column=project.target_column,
                    target_positive_label=str(updates["target_positive_label"]),
                    problem_type=project.problem_type,
                )
                report.raise_if_blocking()

        for field, value in updates.items():
            if field in allowed_fields:
                setattr(project, field, value)
        
        project.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(project)
        return project
    
    def delete_project(self, project_id: str) -> bool:
        """Soft delete project."""
        project = self.get_project(project_id)
        if not project:
            return False
        project.is_active = False
        self.db.commit()
        return True
    
    # =========================================================================
    # Model Training
    # =========================================================================
    
    def train_model(self, project_id: str) -> TrainedModel:
        """Train ML model for project with train/test split."""
        from sklearn.model_selection import train_test_split
        from app.ml.router import route_training
        from app.ml.model_loader import build_model_for_strategy, write_route_meta
        from app.ml.explainers.shap_explainer import SHAPExplainer
        
        project = self.get_project(project_id)
        if not project:
            raise ValueError("Project not found")
        
        project.status = "training"
        self.db.commit()
        
        is_regression = project.problem_type == "regression"
        
        try:
            # Load data
            dataset_service = DatasetService(self.db, self.org_id)
            df = dataset_service.load_dataframe(project.dataset_id)

            # Deterministic hygiene gate (same rules as upload / create)
            profile = profile_dataframe(
                df,
                target_column=project.target_column,
                target_positive_label=(
                    None
                    if is_regression
                    else str(project.target_positive_label)
                ),
                problem_type=project.problem_type,
                feature_columns=list(project.feature_columns or []),
            )
            cleaned_features = apply_feature_exclusions(
                list(project.feature_columns or []), profile
            )
            if cleaned_features != list(project.feature_columns or []):
                project.feature_columns = cleaned_features
            if not cleaned_features:
                raise ProfilingError(
                    "No usable feature columns left after excluding identifiers/constants.",
                    code="no_usable_features",
                    blocking_issues=[{"code": "no_usable_features"}],
                    warnings=profile.warnings,
                )
            profile.raise_if_blocking()

            # Prepare features and target (raw; encode only after split)
            X_raw = df[project.feature_columns].copy()
            raw_target = df[project.target_column].copy()
            y = raw_target.copy()
            
            if is_regression:
                # For regression, ensure target is numeric
                y = pd.to_numeric(y, errors='coerce')
                y = y.fillna(y.median())
            else:
                # Convert target to binary for classification
                target_values = y.astype(str)
                unique_values = sorted(target_values.dropna().unique().tolist())
                if len(unique_values) < 2:
                    raise ProfilingError(
                        f"Target column '{project.target_column}' must contain at least two classes. "
                        f"Found: {unique_values}",
                        code="target_single_class",
                        present_target_values=unique_values,
                    )
                positive_label = str(project.target_positive_label)
                if positive_label not in unique_values:
                    raise ProfilingError(
                        f"Positive label '{positive_label}' does not appear in target column "
                        f"'{project.target_column}'. Found values: {unique_values}. "
                        "Set target_positive_label to a value that exists in the data, then retrain.",
                        code="positive_label_not_in_data",
                        present_target_values=unique_values,
                        blocking_issues=[
                            {
                                "code": "positive_label_not_in_data",
                                "positive_label": positive_label,
                                "present_target_values": unique_values,
                            }
                        ],
                    )
                y = (target_values == positive_label).astype(int)
                class_counts = y.value_counts()
                if class_counts.min() < 2:
                    raise ValueError(
                        "Training requires at least 2 samples in each class. "
                        f"Class distribution: {class_counts.to_dict()}"
                    )
            
            # Phase 1.5: train / calib / test split, then fit encoding on train only
            from app.ml.pipelines.feature_pipeline import FeatureTransformer

            stratify_labels = y if not is_regression else None
            X_temp, X_test_raw, y_temp, y_test, temp_idx, test_idx = train_test_split(
                X_raw,
                y,
                df.index,
                test_size=settings.test_size,
                random_state=42,
                stratify=stratify_labels,
            )
            # calib share of remaining pool
            calib_frac = min(0.5, max(0.05, settings.calib_size / max(1e-6, (1.0 - settings.test_size))))
            try:
                X_train_raw, X_calib_raw, y_train, y_calib, train_idx, calib_idx = train_test_split(
                    X_temp,
                    y_temp,
                    temp_idx,
                    test_size=calib_frac,
                    random_state=42,
                    stratify=y_temp if not is_regression else None,
                )
            except ValueError:
                # tiny / thin class structure
                X_train_raw, X_calib_raw, y_train, y_calib, train_idx, calib_idx = train_test_split(
                    X_temp,
                    y_temp,
                    temp_idx,
                    test_size=0.25,
                    random_state=42,
                )

            if not is_regression:
                for name, ys in (("train", y_train), ("calib", y_calib), ("test", y_test)):
                    if len(np.unique(ys)) < 2:
                        raise ValueError(
                            f"Split '{name}' has a single class. Use more balanced data or a larger dataset."
                        )

            transformer = FeatureTransformer(
                drop_leakage=settings.drop_leakage_columns,
                add_missing_indicators=settings.add_missing_indicators,
            )
            # Do not protect feature_columns wholesale — ID columns must still drop.
            # (IDs are never protectable; only non-ID selections get through.)
            X_train = transformer.fit_transform(
                X_train_raw,
                y_train,
                protected_columns=None,
            )
            X_calib = transformer.transform(X_calib_raw)
            X_test = transformer.transform(X_test_raw)
            # Preserve B1 intent breadcrumb when replacing config after fit
            prior_intent = None
            if isinstance(project.feature_config, dict):
                prior_intent = project.feature_config.get("intent")
            new_cfg = transformer.to_feature_config() or {}
            if prior_intent and isinstance(new_cfg, dict):
                new_cfg = {**new_cfg, "intent": prior_intent}
            project.feature_config = new_cfg
            if transformer.dropped_columns:
                print(f"Dropped feature columns: {transformer.dropped_reasons}")
                # Keep project.feature_columns aligned with what scoring requires —
                # constants/IDs dropped at fit must not remain "required" at predict.
                kept = [
                    c
                    for c in (project.feature_columns or [])
                    if c not in set(transformer.dropped_columns)
                ]
                if kept:
                    project.feature_columns = kept

            # Phase 1b: route foundation (small/medium) vs stacked ensemble (large)
            force = None
            if settings.routing_mode in ("foundation_model", "ensemble"):
                force = settings.routing_mode
            decision = route_training(
                X_train,
                max_foundation_rows=settings.foundation_max_rows,
                max_foundation_features=settings.foundation_max_features,
                force_strategy=force,
            )
            print(f"Model routing: {decision.strategy} — {decision.reason}")

            start_time = time.time()
            model = build_model_for_strategy(
                decision.strategy,
                problem_type=project.problem_type,
            )
            train_kwargs = {
                "validation_data": (X_test, y_test),
                "calibration_data": (X_calib, y_calib),
            }
            try:
                metrics = model.train(X_train, y_train, **train_kwargs)
            except TypeError:
                # Older model signature without calibration_data
                metrics = model.train(X_train, y_train, validation_data=(X_test, y_test))
            training_time = time.time() - start_time

            # Per-project decision threshold (tuned on calib; default 0.5 if unset)
            if not is_regression:
                calib_proba = np.asarray(model.predict_proba(X_calib), dtype=float).ravel()
                pos_rate = float(y_calib.mean())
                thr_metric = (
                    "accuracy"
                    if pos_rate < 0.35 or pos_rate > 0.65
                    else "balanced_accuracy"
                )
                thr, thr_meta = tune_decision_threshold(
                    y_calib.to_numpy(),
                    calib_proba,
                    metric=thr_metric,
                )
                project.feature_config = set_decision_threshold_meta(
                    project.feature_config,
                    thr,
                    metric=thr_meta["metric"],
                    metric_value=thr_meta["metric_value"],
                    tuned_on="calibration",
                )
            
            # Save model
            version = f"v{len(project.trained_models) + 1}"
            model_path = os.path.join(self.model_dir, project.id, version)
            os.makedirs(model_path, exist_ok=True)
            model.save(model_path)
            transformer.save(model_path)
            write_route_meta(
                model_path,
                decision.strategy,
                reason=decision.reason,
                extra={
                    **decision.to_dict(),
                    "backend": getattr(model, "backend", "ensemble"),
                    "n_train": len(X_train),
                    "n_calib": len(X_calib),
                    "n_test": len(X_test),
                    "metrics_snapshot": {
                        k: metrics.get(k)
                        for k in (
                            "ensemble_auc_roc",
                            "ensemble_accuracy",
                            "ensemble_brier",
                            "conformal_quantile",
                            "conformal_mean_width_calib",
                            "prob_calibration_fitted",
                        )
                        if k in metrics
                    },
                },
            )
            
            # Save training data for SHAP
            X_train.to_parquet(os.path.join(model_path, "training_data.parquet"))
            
            # Save test set with original values for prediction UI
            test_df = df.iloc[test_idx].copy()
            test_df.to_parquet(os.path.join(model_path, "test_data.parquet"))
            
            # Calculate feature importance
            try:
                primary = model.get_primary_model() if hasattr(model, "get_primary_model") else model
                explainer = SHAPExplainer(primary, background_data=X_train)
                global_result = explainer.explain_global(X_train)
                importance = global_result.get("feature_importance", {})
            except Exception as shap_err:
                # Non-fatal: training still succeeds without SHAP importance
                try:
                    importance = model.get_feature_importance()
                except Exception:
                    importance = {}
                print(f"SHAP global importance skipped: {shap_err}")
            
            # Deactivate previous models
            self.db.query(TrainedModel).filter(
                TrainedModel.project_id == project_id,
                TrainedModel.is_active == True
            ).update({"is_active": False})
            
            # Create trained model record with appropriate metrics
            if is_regression:
                trained_model = TrainedModel(
                    project_id=project_id,
                    version=version,
                    model_path=model_path,
                    # Regression metrics
                    mae=metrics.get("ensemble_mae"),
                    mse=metrics.get("ensemble_mse"),
                    rmse=metrics.get("ensemble_rmse"),
                    r2_score=metrics.get("ensemble_r2_score"),
                    feature_importance=importance,
                    training_samples=len(X_train),
                    training_time_seconds=training_time,
                    is_active=True,
                )
            else:
                trained_model = TrainedModel(
                    project_id=project_id,
                    version=version,
                    model_path=model_path,
                    # Classification metrics (held-out test)
                    accuracy=metrics.get("ensemble_accuracy"),
                    precision_score=metrics.get("ensemble_precision"),
                    recall_score=metrics.get("ensemble_recall"),
                    f1_score=metrics.get("ensemble_f1_score"),
                    auc_roc=metrics.get("ensemble_auc_roc"),
                    feature_importance=importance,
                    training_samples=len(X_train),
                    training_time_seconds=training_time,
                    is_active=True,
                )
            
            self.db.add(trained_model)
            
            # Canonical ready status (accept "trained" on read for older DBs)
            project.status = "ready"
            self.db.commit()
            self.db.refresh(trained_model)
            
            return trained_model
            
        except ProfilingError:
            project.status = "error"
            self.db.commit()
            raise
        except Exception as e:
            project.status = "error"
            self.db.commit()
            raise ValueError(f"Training failed: {str(e)}")
    
    def get_test_data(self, project_id: str, limit: int = 50) -> list[dict]:
        """Get test set rows for a trained project."""
        project = self.get_project(project_id)
        if not project or project.status not in ["trained", "ready"]:
            raise ValueError("Project not trained")
        
        trained_model = self.get_active_model(project_id)
        if not trained_model:
            raise ValueError("No trained model available")
        
        test_path = os.path.join(trained_model.model_path, "test_data.parquet")
        if not os.path.exists(test_path):
            # Fallback: return sample from dataset
            dataset_service = DatasetService(self.db, self.org_id)
            df = dataset_service.load_dataframe(project.dataset_id)
            return df.head(limit).to_dict(orient='records')
        
        test_df = pd.read_parquet(test_path)
        return test_df.head(limit).to_dict(orient='records')
    
    def get_active_model(self, project_id: str) -> Optional[TrainedModel]:
        """Get currently active model for project."""
        return self.db.query(TrainedModel).filter(
            TrainedModel.project_id == project_id,
            TrainedModel.is_active == True
        ).first()
    
    # =========================================================================
    # Predictions
    # =========================================================================
    
    def predict(
        self,
        project_id: str,
        features: dict[str, Any],
        entity_id: Optional[str] = None,
        include_explanations: bool = True,
        include_recommendations: bool = True,
        *,
        persist: bool = True,
        source: str = "api",
    ) -> dict[str, Any]:
        """Make prediction for a project.

        Args:
            persist: When False (what-if/sim), do not write ProjectPrediction rows.
            source: Provenance tag for response/metadata (api|batch|simulation).
        """
        from app.ml.model_loader import load_routed_model, detect_strategy
        from app.ml.explainers.case_explainer import CaseExplainer
        from app.ml.feature_validation import validate_required_features
        
        project = self.get_project(project_id)
        if not project or project.status not in ["trained", "ready"]:
            raise ValueError("Project not found or not trained")
        
        trained_model = self.get_active_model(project_id)
        if not trained_model:
            raise ValueError("No trained model available")

        # P0: reject incomplete inputs (no silent zero-impute scoring)
        validate_required_features(features, project.feature_columns or [])
        
        is_regression = project.problem_type == "regression"
        
        # Load model (foundation or ensemble per route_meta)
        model = load_routed_model(
            trained_model.model_path,
            problem_type=project.problem_type,
        )
        routing_strategy = detect_strategy(trained_model.model_path)

        # Prefer train-safe FeatureTransformer; fallback to feature_config
        from app.ml.pipelines.feature_pipeline import FeatureTransformer

        ft_path = os.path.join(trained_model.model_path, "feature_transformer.joblib")
        if os.path.exists(ft_path):
            transformer = FeatureTransformer()
            transformer.load(trained_model.model_path)
            raw_df = pd.DataFrame([{c: features.get(c) for c in project.feature_columns}])
            feature_df = transformer.transform(raw_df)
        else:
            feature_df = pd.DataFrame([features])
            if project.feature_config:
                for col, config in project.feature_config.items():
                    if col.startswith("_"):
                        continue
                    if col in feature_df.columns and isinstance(config, dict):
                        if config.get("type") == "categorical":
                            categories = config.get("categories", [])
                            value = feature_df[col].iloc[0]
                            code_map = config.get("code_map")
                            if code_map and str(value) in code_map:
                                feature_df[col] = code_map[str(value)]
                            elif value in categories:
                                feature_df[col] = categories.index(value)
                            else:
                                feature_df[col] = -1
            for col in project.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            # align to model feature names if present
            if getattr(model, "feature_names", None):
                for col in model.feature_names:
                    if col not in feature_df.columns:
                        feature_df[col] = 0
                feature_df = feature_df[model.feature_names]
            else:
                feature_df = feature_df[[c for c in project.feature_columns if c in feature_df.columns]]
        
        # Point estimate + conformal interval + abstention (Phase 1a)
        uncertainty = model.predict_with_uncertainty(feature_df)[0]
        confidence_interval = uncertainty.as_interval_dict()
        low_confidence = uncertainty.low_confidence
        abstention_reason = uncertainty.abstention_reason
        confidence = float(model.get_confidence(feature_df)[0])
        if low_confidence:
            confidence = min(confidence, 0.5)

        if is_regression:
            predicted_value = float(uncertainty.prediction)
            result = {
                "predicted_value": predicted_value,
                "confidence": confidence,
                "target": project.target_description,
                "problem_type": "regression",
                "confidence_interval": confidence_interval,
                "low_confidence": low_confidence,
                "abstention_reason": abstention_reason,
                "model_disagreement": uncertainty.disagreement,
                "routing_strategy": routing_strategy,
                "model_backend": getattr(model, "backend", "ensemble"),
            }
        else:
            probability = float(uncertainty.prediction)
            risk_level = self._get_risk_level(probability)
            result = {
                "probability": probability,
                "confidence": confidence,
                "risk_level": risk_level,
                "target": project.target_description,
                "routing_strategy": routing_strategy,
                "model_backend": getattr(model, "backend", "ensemble"),
                "problem_type": "classification",
                "confidence_interval": confidence_interval,
                "low_confidence": low_confidence,
                "abstention_reason": abstention_reason,
                "model_disagreement": uncertainty.disagreement,
            }
        
        # Phase 2: SHAP + LIME + Explanation Consistency on routed model
        shap_values = None
        top_factors = None
        if include_explanations and not is_regression:
            try:
                training_path = os.path.join(trained_model.model_path, "training_data.parquet")
                if not os.path.exists(training_path):
                    raise FileNotFoundError("training_data.parquet missing — retrain project")
                training_data = pd.read_parquet(training_path)
                # Align background to model feature names
                if getattr(model, "feature_names", None):
                    for col in model.feature_names:
                        if col not in training_data.columns:
                            training_data[col] = 0
                    training_data = training_data[model.feature_names]

                outcome = (
                    project.target_description
                    or project.target_column
                    or "the outcome"
                )
                case = CaseExplainer().explain(
                    model,
                    feature_df,
                    training_data,
                    raw_features=features,
                    outcome_label=str(outcome).lower(),
                    positive_label=str(project.target_positive_label or "Yes"),
                    negative_label="No",
                    top_k=5,
                )
                top_factors = case.get("top_factors") or []
                shap_values = {
                    f["feature"]: f.get("impact")
                    for f in (case.get("shap") or {}).get("top_features") or []
                }
                result["explanations"] = {
                    "shap": case.get("shap"),
                    "lime": case.get("lime"),
                    "consistency": case.get("consistency"),
                    "drivers": case.get("drivers"),
                    "all_factors": case.get("all_factors"),
                    "methods_available": case.get("methods_available"),
                    "degraded": case.get("degraded"),
                    "errors": case.get("errors"),
                }
                result["explanation_consistency"] = case.get("consistency")

                # B4: flag confounded / non-intervenable drivers (scrutiny, not causal fact)
                try:
                    from app.ml.blindspot import annotate_drivers, detect_blindspots

                    blindspots = detect_blindspots(
                        top_factors=top_factors,
                        features=features,
                        feature_config=project.feature_config,
                        consistency=case.get("consistency"),
                        training_data=training_data,
                        target_column=project.target_column,
                        target_positive_label=project.target_positive_label,
                        outcome_label=str(outcome),
                    )
                    result["blindspots"] = blindspots
                    result["blindspot_warnings"] = blindspots.get("warnings") or []
                    if result["explanations"].get("drivers"):
                        result["explanations"]["drivers"] = annotate_drivers(
                            result["explanations"]["drivers"], blindspots
                        )
                    top_factors = annotate_drivers(top_factors, blindspots)
                except Exception:
                    result["blindspot_warnings"] = []
            except Exception as e:
                result["explanations"] = {"error": str(e), "degraded": True}
        
        # Phase 3: business insight brief from drivers + trust context
        insights = []
        insight_brief = None
        if top_factors and include_explanations and not is_regression:
            from app.insights.nlp_generator import InsightGenerator
            from app.ml.soft_range import interval_is_soft

            soft = interval_is_soft(
                point=float(result.get("probability", 0.5)),
                lower=float(confidence_interval.get("lower", 0)),
                upper=float(confidence_interval.get("upper", 1)),
                low_confidence=bool(low_confidence),
                is_regression=False,
            )
            outcome = (
                project.target_description
                or project.target_column
                or "the outcome"
            )
            brief = InsightGenerator().generate_case_insights(
                drivers=top_factors,
                probability=float(result.get("probability", 0.5)),
                features=features,
                outcome_label=str(outcome),
                consistency=result.get("explanation_consistency"),
                low_confidence=bool(low_confidence),
                soft_range=bool(soft.get("is_soft")),
            )
            insight_brief = brief
            insights = brief.get("insights") or []
            result["insights"] = insights
            action_context = brief.get("action_context") or {}
            # Soft-rerank primary lever away from blindspot-flagged drivers
            preferred = (result.get("blindspots") or {}).get("preferred_primary_feature")
            if preferred and action_context.get("primary_lever"):
                cur = (action_context.get("primary_lever") or {}).get("feature")
                if cur and cur != preferred:
                    alt = next(
                        (d for d in top_factors if d.get("feature") == preferred),
                        None,
                    )
                    if alt:
                        from app.insights.feature_mapping import get_action_hint, get_feature_info

                        info = get_feature_info(preferred)
                        action_context = {
                            **action_context,
                            "primary_lever": {
                                "feature": preferred,
                                "display_name": info.get("display_name") or preferred,
                                "suggestion": get_action_hint(preferred, True),
                            },
                            "blindspot_reranked": True,
                            "previous_primary_feature": cur,
                        }
            result["insight_brief"] = {
                "headline": brief.get("headline"),
                "summary": brief.get("summary"),
                "theme_rollup": brief.get("theme_rollup"),
                "risk_factors": brief.get("risk_factors"),
                "protective_factors": brief.get("protective_factors"),
                "overall_severity": brief.get("overall_severity"),
                "trust_note": brief.get("trust_note"),
                "action_context": action_context,
            }
        
        # Get recommendations (only for classification)
        recommendations = None
        if include_recommendations and not is_regression:
            from app.ml.soft_range import interval_is_soft as _iis

            soft_info = _iis(
                point=float(result.get("probability", 0.5)),
                lower=float(confidence_interval.get("lower", 0)),
                upper=float(confidence_interval.get("upper", 1)),
                low_confidence=bool(low_confidence),
                is_regression=False,
            )
            cons = result.get("explanation_consistency") or {}
            # A7: blend historical action success into impact when available
            effectiveness = self.get_action_effectiveness(project_id)
            recommendations = self._get_recommendations(
                project_id,
                result.get("probability", 0.5),
                features,
                top_factors,
                action_context=(insight_brief or {}).get("action_context")
                if insight_brief
                else None,
                soft_case=bool(soft_info.get("is_soft")),
                low_confidence=bool(low_confidence),
                consistency_trust=cons.get("trust_level"),
                outcome_label=str(
                    project.target_description
                    or project.target_column
                    or "the outcome"
                ),
                effectiveness_data=effectiveness or None,
                project=project,
            )
            # Extract decision summary stashed on first rec
            if recommendations:
                summary = recommendations[0].pop("_decision_summary", None)
                scoring_meta = recommendations[0].pop("_scoring", None)
                for r in recommendations[1:]:
                    r.pop("_decision_summary", None)
                    r.pop("_scoring", None)
                if summary:
                    result["decision_summary"] = summary
                if scoring_meta:
                    result["recommendation_scoring"] = scoring_meta
            result["recommendations"] = recommendations
        
        result["source"] = source
        result["persisted"] = False
        result["prediction_id"] = None
        if entity_id:
            result["entity_id"] = entity_id

        # Store prediction only for real (non-simulation) scoring
        if persist:
            safe_features = _json_safe(features)
            safe_shap = _json_safe(shap_values) if shap_values is not None else None
            trust_meta = {
                "confidence_interval": _json_safe(result.get("confidence_interval")),
                "abstention_reason": result.get("abstention_reason"),
                "model_disagreement": result.get("model_disagreement"),
                "low_confidence": bool(low_confidence),
            }
            # Keep interval/trust on the row so Don't-act hydrate can rebuild the spine
            if isinstance(safe_shap, dict):
                safe_shap = {**safe_shap, "_knowa_trust": trust_meta}
            elif safe_shap is None:
                safe_shap = {"_knowa_trust": trust_meta}
            else:
                safe_shap = {"_values": safe_shap, "_knowa_trust": trust_meta}
            safe_top = _json_safe(top_factors) if top_factors is not None else None
            safe_recs = _json_safe(recommendations) if recommendations is not None else None
            if is_regression:
                prediction = ProjectPrediction(
                    project_id=project_id,
                    model_version=trained_model.version,
                    entity_id=entity_id,
                    features=safe_features,
                    predicted_value=float(predicted_value) if predicted_value is not None else None,
                    confidence=float(confidence) if confidence is not None else None,
                    shap_values=safe_shap,
                    top_factors=safe_top,
                    low_confidence=bool(low_confidence),
                )
            else:
                prediction = ProjectPrediction(
                    project_id=project_id,
                    model_version=trained_model.version,
                    entity_id=entity_id,
                    features=safe_features,
                    probability=float(result["probability"]) if result.get("probability") is not None else None,
                    confidence=float(confidence) if confidence is not None else None,
                    risk_level=result["risk_level"],
                    shap_values=safe_shap,
                    top_factors=safe_top,
                    recommendations=safe_recs,
                    low_confidence=bool(low_confidence),
                )
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            result["prediction_id"] = prediction.id
            result["persisted"] = True
        
        return result

    def predict_batch(
        self,
        project_id: str,
        rows: list[dict[str, Any]],
        *,
        entity_ids: Optional[list[Optional[str]]] = None,
        max_rows: int = 1000,
    ) -> dict[str, Any]:
        """
        Score many cases for triage: probability + risk + soft flag only.

        Skips SHAP/LIME, insights, and recommendations for speed.
        Does not persist rows (batch triage is ephemeral by default).
        """
        from app.ml.model_loader import load_routed_model, detect_strategy
        from app.ml.feature_validation import (
            FeatureValidationError,
            validate_required_features,
            resolve_entity_id,
        )
        from app.ml.soft_range import interval_is_soft, action_tier

        project = self.get_project(project_id)
        if not project or project.status not in ["trained", "ready"]:
            raise ValueError("Project not found or not trained")
        trained_model = self.get_active_model(project_id)
        if not trained_model:
            raise ValueError("No trained model available")
        if not rows:
            raise ValueError("rows must be a non-empty list")
        if len(rows) > max_rows:
            raise ValueError(f"Batch size {len(rows)} exceeds max_rows={max_rows}")

        is_regression = project.problem_type == "regression"
        required = list(project.feature_columns or [])
        routing = detect_strategy(trained_model.model_path)
        # ensure model artifacts load once (predict reuses cached path via disk)
        load_routed_model(trained_model.model_path, problem_type=project.problem_type)

        results: list[dict[str, Any]] = []
        n_ok = 0
        n_err = 0

        for i, raw in enumerate(rows):
            raw = raw or {}
            # Allow full dataset rows (id + features + target); strip to features
            features = {c: raw.get(c) for c in required}
            eid = None
            if entity_ids and i < len(entity_ids):
                eid = entity_ids[i]
            if not eid:
                eid = resolve_entity_id(raw, row_index=i)

            try:
                validate_required_features(features, required)
            except FeatureValidationError as e:
                n_err += 1
                results.append({
                    "index": i,
                    "entity_id": eid,
                    "error": e.as_detail(),
                    "ok": False,
                })
                continue

            try:
                # reuse single-path scoring without explain/rec/persist
                out = self.predict(
                    project_id,
                    features,
                    entity_id=eid,
                    include_explanations=False,
                    include_recommendations=False,
                    persist=False,
                    source="batch",
                )
                compact: dict[str, Any] = {
                    "index": i,
                    "entity_id": eid,
                    "ok": True,
                    "risk_level": out.get("risk_level"),
                    "confidence": out.get("confidence"),
                    "confidence_interval": out.get("confidence_interval"),
                    "low_confidence": out.get("low_confidence"),
                    "routing_strategy": out.get("routing_strategy") or routing,
                }
                if is_regression:
                    compact["predicted_value"] = out.get("predicted_value")
                    compact["soft_case"] = bool(out.get("low_confidence"))
                else:
                    p = float(out.get("probability") or 0)
                    ci = out.get("confidence_interval") or {}
                    soft = interval_is_soft(
                        point=p,
                        lower=float(ci.get("lower", 0)),
                        upper=float(ci.get("upper", 1)),
                        low_confidence=bool(out.get("low_confidence")),
                        is_regression=False,
                    )
                    compact["probability"] = p
                    compact["soft_case"] = bool(soft.get("is_soft"))
                    compact["soft_reason"] = soft.get("reason")
                    compact["action_tier"] = action_tier(p)
                n_ok += 1
                results.append(compact)
            except Exception as e:
                n_err += 1
                results.append({
                    "index": i,
                    "entity_id": eid,
                    "ok": False,
                    "error": {"code": "score_failed", "message": str(e)},
                })

        # Sort ok classification rows by probability desc for triage convenience
        ranked = [r for r in results if r.get("ok") and "probability" in r]
        ranked.sort(key=lambda r: float(r.get("probability") or 0), reverse=True)
        for rank, r in enumerate(ranked, start=1):
            r["priority_rank"] = rank

        return {
            "project_id": project_id,
            "n_requested": len(rows),
            "n_scored": n_ok,
            "n_errors": n_err,
            "include_explanations": False,
            "include_recommendations": False,
            "persisted": False,
            "note": (
                "Batch triage scores only — use single-case /predict with "
                "explanations for full drivers and recommendations."
            ),
            "results": results,
        }

    def _prepare_feature_frame(
        self,
        project: Project,
        trained_model: TrainedModel,
        features: dict[str, Any],
        model,
    ) -> pd.DataFrame:
        """Encode a raw feature dict using saved FeatureTransformer or legacy config."""
        from app.ml.pipelines.feature_pipeline import FeatureTransformer

        ft_path = os.path.join(trained_model.model_path, "feature_transformer.joblib")
        if os.path.exists(ft_path):
            transformer = FeatureTransformer()
            transformer.load(trained_model.model_path)
            raw_df = pd.DataFrame([{c: features.get(c) for c in (project.feature_columns or [])}])
            return transformer.transform(raw_df)

        feature_df = pd.DataFrame([features])
        if project.feature_config:
            for col, config in project.feature_config.items():
                if col.startswith("_") or not isinstance(config, dict):
                    continue
                if col in feature_df.columns and config.get("type") == "categorical":
                    categories = config.get("categories", [])
                    value = feature_df[col].iloc[0]
                    code_map = config.get("code_map")
                    if code_map and str(value) in code_map:
                        feature_df[col] = code_map[str(value)]
                    elif value in categories:
                        feature_df[col] = categories.index(value)
                    else:
                        feature_df[col] = -1
        for col in project.feature_columns or []:
            if col not in feature_df.columns:
                feature_df[col] = 0
        if getattr(model, "feature_names", None):
            for col in model.feature_names:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            return feature_df[model.feature_names]
        return feature_df[[c for c in (project.feature_columns or []) if c in feature_df.columns]]

    def spot_check(self, project_id: str, limit: int = 50) -> dict[str, Any]:
        """
        Compare held-out test labels to model predictions (classification).

        Returns plain-language verification stats for the project UI.
        """
        project = self.get_project(project_id)
        if not project or project.status not in ("trained", "ready"):
            raise ValueError("Project not found or not trained")
        if project.problem_type == "regression":
            return {
                "problem_type": "regression",
                "supported": False,
                "message": "Spot-check against Yes/No outcomes is for classification projects.",
            }

        trained_model = self.get_active_model(project_id)
        if not trained_model:
            raise ValueError("No trained model available")

        from app.ml.model_loader import load_routed_model

        model = load_routed_model(trained_model.model_path, problem_type=project.problem_type)
        rows = self.get_test_data(project_id, limit=limit)
        if not rows:
            return {
                "supported": True,
                "n": 0,
                "message": "No held-out rows available yet. Prepare the project again.",
            }

        target = project.target_column
        pos = str(project.target_positive_label)
        y_true: list[int] = []
        y_prob: list[float] = []
        y_pred: list[int] = []
        soft_count = 0

        for row in rows:
            if target not in row:
                continue
            actual = 1 if str(row[target]) == pos else 0
            features = {c: row.get(c) for c in (project.feature_columns or [])}
            try:
                feat_df = self._prepare_feature_frame(project, trained_model, features, model)
                uncertainty = model.predict_with_uncertainty(feat_df)[0]
                p = float(uncertainty.prediction)
                from app.ml.soft_range import interval_is_soft

                soft_info = interval_is_soft(
                    point=p,
                    lower=float(uncertainty.lower),
                    upper=float(uncertainty.upper),
                    low_confidence=bool(uncertainty.low_confidence),
                    is_regression=False,
                )
                if soft_info["is_soft"]:
                    soft_count += 1
                # also track reasons optional later
                _ = float(uncertainty.interval_width)
            except Exception:
                continue
            y_true.append(actual)
            y_prob.append(p)
            thr = get_decision_threshold(project)
            y_pred.append(1 if p >= thr else 0)

        n = len(y_true)
        if n == 0:
            return {
                "supported": True,
                "n": 0,
                "message": "Could not score held-out rows. Try retrain.",
            }

        y_true_a = np.asarray(y_true)
        y_pred_a = np.asarray(y_pred)
        y_prob_a = np.asarray(y_prob)

        agree = int((y_true_a == y_pred_a).sum())
        agree_rate = agree / n

        # High-risk bucket (p >= 0.6): among flagged, what fraction were true Yes
        high_mask = y_prob_a >= 0.6
        n_high = int(high_mask.sum())
        high_true = int(y_true_a[high_mask].sum()) if n_high else 0
        high_precision = (high_true / n_high) if n_high else None

        # Low-risk bucket (p < 0.4): among calm, what fraction were true No
        low_mask = y_prob_a < 0.4
        n_low = int(low_mask.sum())
        low_true_neg = int((y_true_a[low_mask] == 0).sum()) if n_low else 0
        low_negative_rate = (low_true_neg / n_low) if n_low else None

        n_pos = int(y_true_a.sum())
        # Among true Yes, how many called high (p>=0.6)
        if n_pos > 0:
            recall_high = float(((y_true_a == 1) & high_mask).sum() / n_pos)
        else:
            recall_high = None

        # Simple grading for business language
        if agree_rate >= 0.75 and (high_precision is None or high_precision >= 0.5):
            grade = "solid"
            grade_label = "Solid compass"
            grade_detail = "Ranking and Yes/No calls track known outcomes well enough for prioritization."
        elif agree_rate >= 0.6:
            grade = "useful"
            grade_label = "Useful guide"
            grade_detail = "Often right on average — still verify drivers on misses before big spends."
        else:
            grade = "rough"
            grade_label = "Rough guide"
            grade_detail = "Known outcomes disagree often. Fix data/labels or retrain before ops use."

        return {
            "supported": True,
            "problem_type": "classification",
            "n": n,
            "agree_count": agree,
            "agree_rate": round(agree_rate, 4),
            "actual_positive": n_pos,
            "actual_negative": n - n_pos,
            "flagged_high": n_high,
            "high_risk_were_true_yes": high_true,
            "high_risk_precision": round(high_precision, 4) if high_precision is not None else None,
            "calm_low": n_low,
            "low_risk_were_true_no": low_true_neg,
            "low_risk_true_negative_rate": (
                round(low_negative_rate, 4) if low_negative_rate is not None else None
            ),
            "true_yes_caught_as_high": (
                round(recall_high, 4) if recall_high is not None else None
            ),
            "soft_signal_share": round(soft_count / n, 4),
            "soft_definition": (
                "Soft when the score sits mid-priority, the residual band is "
                "nearly open on 0–100%, or the uncertainty gate fires — not "
                "because a fat residual bar always reaches both ends."
            ),
            "decision_threshold": get_decision_threshold(project),
            "high_threshold": 0.6,
            "low_threshold": 0.4,
            "grade": grade,
            "grade_label": grade_label,
            "grade_detail": grade_detail,
            "plain_summary": (
                f"On {n} held-out people with known outcomes, the Yes/No call matched "
                f"{agree} times ({agree_rate:.0%}). "
                + (
                    f"Of {n_high} flagged high-risk, {high_true} had actually Yes."
                    if n_high
                    else "Few/no high-risk flags in this sample."
                )
            ),
        }

    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level."""
        if probability >= 0.8:
            return "critical"
        elif probability >= 0.6:
            return "high"
        elif probability >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_recommendations(
        self,
        project_id: str,
        probability: float,
        features: dict,
        top_factors: Optional[list],
        action_context: Optional[dict] = None,
        *,
        soft_case: bool = False,
        low_confidence: bool = False,
        consistency_trust: Optional[str] = None,
        outcome_label: str = "the outcome",
        effectiveness_data: Optional[dict] = None,
        project: Optional[Project] = None,
    ) -> list[dict]:
        """
        Phase 4: hybrid decision scoring.

        Uses domain action catalog (Telco default; HR attrition when detected)
        + org custom actions, ranked by final_score = α·impact + β·(1−cost) + γ·relevance,
        with Phase-3 action_context boosting relevance. A7 effectiveness_data can
        gently recalibrate impact for previously logged actions.
        """
        from app.recommendations.decision_scorer import DecisionScorer
        from app.recommendations.action_catalog import action_from_custom
        from app.recommendations.domains import detect_domain

        proj = project or self.get_project(project_id)
        domain = detect_domain(
            feature_columns=getattr(proj, "feature_columns", None) if proj else None,
            features=features,
            project_name=getattr(proj, "name", None) if proj else None,
            target_column=getattr(proj, "target_column", None) if proj else None,
            target_description=getattr(proj, "target_description", None) if proj else None,
        )

        custom_rows = self.db.query(CustomAction).filter(
            CustomAction.organization_id == self.org_id,
            CustomAction.is_active == True,
        ).all()
        custom_actions = [
            action_from_custom(
                code=a.code,
                name=a.name,
                description=a.description or "",
                estimated_cost=float(a.estimated_cost if a.estimated_cost is not None else 0),
                estimated_impact=float(
                    a.estimated_impact if a.estimated_impact is not None else 0.5
                ),
                applicable_when=a.applicable_when,
            )
            for a in custom_rows
        ]

        scorer = DecisionScorer(effectiveness_data=effectiveness_data)
        result = scorer.score_case(
            features=features or {},
            probability=float(probability),
            top_factors=top_factors or [],
            action_context=action_context,
            custom_actions=custom_actions,
            soft_case=soft_case,
            low_confidence=low_confidence,
            consistency_trust=consistency_trust,
            outcome_label=outcome_label,
            top_n=5,
            domain=domain,
        )
        recs = result.get("recommendations") or []
        if recs:
            recs[0]["_decision_summary"] = result.get("decision_summary")
            sc = result.get("scoring") or {}
            if effectiveness_data:
                sc = {**sc, "uses_feedback_effectiveness": True, "effectiveness_n_actions": len(effectiveness_data)}
            recs[0]["_scoring"] = sc
        return recs

    def _generate_reasoning(self, action: CustomAction, probability: float, top_factors: Optional[list]) -> str:
        """Legacy helper (custom action free-text)."""
        risk = "high" if probability > 0.6 else "moderate" if probability > 0.4 else "low"
        reasoning = f"With {risk} risk ({probability:.0%}), "
        if top_factors and len(top_factors) > 0:
            top = top_factors[0]
            reasoning += f"'{top['feature']}' is the main driver. "
        reasoning += f"'{action.name}' could help address this."
        return reasoning
    
    def _generate_business_insights(
        self, 
        top_factors: list[dict],
        features: dict[str, Any],
        project: Project,
        probability: float = 0.5,
        consistency: Optional[dict] = None,
        low_confidence: bool = False,
    ) -> list[dict]:
        """Delegate to Phase-3 InsightGenerator (kept for internal reuse)."""
        from app.insights.nlp_generator import InsightGenerator

        outcome = project.target_description or project.target_column or "the outcome"
        brief = InsightGenerator().generate_case_insights(
            drivers=top_factors or [],
            probability=probability,
            features=features or {},
            outcome_label=str(outcome),
            consistency=consistency,
            low_confidence=low_confidence,
        )
        return brief.get("insights") or []
    
    # =========================================================================
    # Simulation
    # =========================================================================

    def _diff_features(
        self,
        base_features: dict[str, Any],
        modified_features: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Only real feature diffs (ignore no-ops and unknown keys)."""
        change_log: list[dict[str, Any]] = []
        for key, after in (modified_features or {}).items():
            before = (base_features or {}).get(key)
            # normalize numpy scalars for comparison
            before_s = _json_safe(before)
            after_s = _json_safe(after)
            if before_s is None and after_s is None:
                continue
            if str(before_s) == str(after_s):
                continue
            change_log.append({
                "feature": key,
                "label": " ".join(
                    w.capitalize()
                    for w in str(key).replace("_", " ").replace("-", " ").split()
                ),
                "before": before_s,
                "after": after_s,
            })
        return change_log

    def _driver_shift(
        self,
        before_factors: Optional[list],
        after_factors: Optional[list],
        top_n: int = 6,
    ) -> list[dict[str, Any]]:
        """Compare SHAP drivers before vs after the scenario."""
        before_map = {
            f.get("feature"): float(f.get("impact") or 0)
            for f in (before_factors or [])
            if f.get("feature")
        }
        after_map = {
            f.get("feature"): float(f.get("impact") or 0)
            for f in (after_factors or [])
            if f.get("feature")
        }
        labels = {
            f.get("feature"): f.get("label") or f.get("feature")
            for f in (before_factors or []) + (after_factors or [])
            if f.get("feature")
        }
        keys = set(before_map) | set(after_map)
        rows = []
        for k in keys:
            b = before_map.get(k, 0.0)
            a = after_map.get(k, 0.0)
            rows.append({
                "feature": k,
                "label": labels.get(k, k),
                "before_impact": round(b, 4),
                "after_impact": round(a, 4),
                "delta": round(a - b, 4),
            })
        rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
        return rows[:top_n]

    def _suggested_tweaks(
        self,
        base_features: dict[str, Any],
        original_pred: dict[str, Any],
        project: Project,
    ) -> list[dict[str, Any]]:
        """Lightweight next-dial suggestions from top risk drivers / primary lever."""
        from app.recommendations.domains import DOMAIN_HR_ATTRITION, detect_domain

        suggestions: list[dict[str, Any]] = []
        brief = original_pred.get("insight_brief") or {}
        lever = (brief.get("action_context") or {}).get("primary_lever") or {}
        factors = original_pred.get("explanations", {}).get("drivers") or original_pred.get(
            "explanations", {}
        ).get("shap", {}).get("top_features") or []

        domain = detect_domain(
            feature_columns=project.feature_columns,
            features=base_features,
            project_name=project.name,
            target_column=project.target_column,
            target_description=project.target_description,
        )

        if lever.get("feature"):
            feat = lever["feature"]
            cur = base_features.get(feat)
            hint = lever.get("suggestion") or lever.get("action_hint")
            if hint:
                suggestions.append({
                    "feature": feat,
                    "label": lever.get("label") or feat,
                    "hint": hint,
                    "current": _json_safe(cur),
                })

        for f in factors[:6]:
            feat = f.get("feature")
            if not feat or feat in {s.get("feature") for s in suggestions}:
                continue
            impact = float(f.get("impact") or 0)
            if impact <= 0:
                continue
            cur = base_features.get(feat)
            alt = None
            hint = f.get("text") or f"Try adjusting {feat} to lower risk pressure."
            fl = str(feat).lower().replace("_", "")

            if domain == DOMAIN_HR_ATTRITION:
                if "overtime" in fl:
                    if str(cur).strip().lower() in ("yes", "y", "true", "1"):
                        alt = "No"
                        hint = "Turn off overtime to test workload as a lever."
                elif "businesstravel" in fl:
                    if "frequent" in str(cur).lower():
                        alt = "Travel_Rarely"
                        hint = "Reduce travel load and re-score attrition risk."
                elif any(x in fl for x in ("jobsatisfaction", "environmentsatisfaction", "relationshipsatisfaction")):
                    try:
                        v = int(float(cur))
                        if v < 4:
                            alt = 4
                            hint = "Job satisfaction scores only run 1–4 here — try the top score."
                    except (TypeError, ValueError):
                        alt = None
                elif "worklifebalance" in fl:
                    try:
                        v = int(float(cur))
                        if v < 4:
                            alt = 4
                            hint = "Work–life balance scores run 1–4 — try raising to 4."
                    except (TypeError, ValueError):
                        alt = None
                elif "distancefromhome" in fl:
                    try:
                        v = float(cur)
                        if v >= 15:
                            alt = max(5, round(v * 0.5, 1))
                            hint = "Proxy for hybrid/remote reducing commute friction."
                    except (TypeError, ValueError):
                        alt = None
                elif "monthlyincome" in fl:
                    try:
                        alt = round(float(cur) * 1.1, 2)
                        hint = "Simulate a ~10% compensation adjustment."
                    except (TypeError, ValueError):
                        alt = None
                elif "yearssincelastpromotion" in fl:
                    try:
                        v = int(float(cur))
                        if v >= 2:
                            alt = 0
                            hint = "Simulate a recent promotion / path reset."
                    except (TypeError, ValueError):
                        alt = None
                elif "stockoptionlevel" in fl:
                    try:
                        if int(float(cur)) <= 0:
                            alt = 1
                            hint = "Simulate granting a stock-option level."
                    except (TypeError, ValueError):
                        alt = None
                elif "trainingtimes" in fl:
                    try:
                        alt = int(float(cur)) + 1
                        hint = "Simulate one additional training cycle."
                    except (TypeError, ValueError):
                        alt = None
            else:
                if "contract" in fl:
                    for cand in ("Two year", "One year", "Two Year"):
                        if str(cur) != cand:
                            alt = cand
                            break
                elif "paperless" in fl:
                    alt = "No" if str(cur) not in ("No", "False", "0") else "Yes"
                elif "charges" in fl or ("monthly" in fl and "income" not in fl):
                    try:
                        alt = round(float(cur) * 0.85, 2)
                    except (TypeError, ValueError):
                        alt = None
                elif "tenure" in fl and "year" not in fl:
                    try:
                        alt = int(float(cur)) + 12
                    except (TypeError, ValueError):
                        alt = None
                elif fl in ("techsupport", "onlinesecurity") and str(cur).lower() in (
                    "no", "false", "0",
                ):
                    alt = "Yes"
                    hint = f"Add {feat} and re-score churn risk."

            if alt is not None:
                suggestions.append({
                    "feature": feat,
                    "label": f.get("label") or feat,
                    "suggested_value": alt,
                    "current": _json_safe(cur),
                    "hint": hint,
                })
            if len(suggestions) >= 4:
                break

        return _json_safe(suggestions)

    @staticmethod
    def _feature_value_bounds(feature: str, domain: str) -> Optional[tuple[float, float]]:
        """Known ordinal ranges so what-if dials stay in-distribution."""
        from app.recommendations.domains import DOMAIN_HR_ATTRITION

        fl = str(feature).lower().replace("_", "").replace("-", "")
        if domain == DOMAIN_HR_ATTRITION:
            if fl in {
                "jobsatisfaction",
                "environmentsatisfaction",
                "relationshipsatisfaction",
                "jobinvolvement",
                "worklifebalance",
                "performancerating",
            }:
                return (1.0, 4.0)
            if fl == "education":
                return (1.0, 5.0)
            if fl == "stockoptionlevel":
                return (0.0, 3.0)
        return None

    def _sanitize_scenario_mods(
        self,
        *,
        modified: dict[str, Any],
        domain: str,
    ) -> tuple[dict[str, Any], list[str]]:
        """Clamp OOD dials (e.g. JobSatisfaction 10 when train range is 1–4)."""
        out: dict[str, Any] = {}
        notes: list[str] = []
        for key, val in (modified or {}).items():
            bounds = self._feature_value_bounds(key, domain)
            if bounds is None:
                out[key] = val
                continue
            lo, hi = bounds
            try:
                num = float(val)
            except (TypeError, ValueError):
                out[key] = val
                continue
            clamped = min(hi, max(lo, num))
            # keep ints for Likert-style fields
            if float(clamped).is_integer():
                clamped_val: Any = int(clamped)
            else:
                clamped_val = clamped
            out[key] = clamped_val
            if abs(clamped - num) > 1e-9:
                label = " ".join(
                    w.capitalize()
                    for w in str(key).replace("_", " ").replace("-", " ").split()
                )
                notes.append(
                    f"{label} only runs from {int(lo)}–{int(hi)} in this dataset; "
                    f"used {clamped_val} instead of {val}."
                )
        return out, notes

    def _rank_tweaks_by_impact(
        self,
        project_id: str,
        base_features: dict[str, Any],
        suggestions: list[dict[str, Any]],
        baseline_prob: float,
        *,
        domain: str,
    ) -> list[dict[str, Any]]:
        """Keep / order dial suggestions by actual score movement for this case."""
        if not suggestions:
            return []
        ranked: list[dict[str, Any]] = []
        for s in suggestions:
            feat = s.get("feature")
            alt = s.get("suggested_value")
            if feat is None or alt is None:
                continue
            safe_mods, _ = self._sanitize_scenario_mods(
                modified={feat: alt}, domain=domain
            )
            alt = safe_mods.get(feat, alt)
            try:
                combined = {**(base_features or {}), feat: alt}
                pred = self.predict(
                    project_id,
                    _json_safe(combined) or {},
                    include_explanations=False,
                    include_recommendations=False,
                    persist=False,
                    source="simulation",
                )
                after = float(pred.get("probability") or pred.get("predicted_value") or 0)
                delta = after - float(baseline_prob)
            except Exception:
                delta = 0.0
                after = float(baseline_prob)
            row = {
                **s,
                "suggested_value": alt,
                "expected_delta": round(delta, 4),
                "expected_probability": round(after, 4),
            }
            ranked.append(row)
        # Prefer levers that actually move the needle (risk down first)
        ranked.sort(key=lambda r: (abs(float(r.get("expected_delta") or 0)),), reverse=True)
        meaningful = [r for r in ranked if abs(float(r.get("expected_delta") or 0)) >= 0.005]
        # Do not invent "suggestions" that leave the score unchanged
        return _json_safe(meaningful)

    def _probe_fallback_levers(
        self,
        project_id: str,
        base_features: dict[str, Any],
        baseline_prob: float,
        *,
        domain: str,
        project: Project,
    ) -> list[dict[str, Any]]:
        """Probe common dials and return those that move this case's score."""
        from app.recommendations.domains import DOMAIN_HR_ATTRITION

        candidates: list[tuple[str, Any, str]] = []
        feats = set(project.feature_columns or [])
        colmap = {str(c).lower(): c for c in feats}

        def has(name: str) -> Optional[str]:
            return colmap.get(name.lower())

        if domain == DOMAIN_HR_ATTRITION:
            ot = has("OverTime")
            if ot and str(base_features.get(ot, "")).lower() in ("yes", "y", "true", "1"):
                candidates.append((ot, "No", "Turn off overtime"))
            elif ot and str(base_features.get(ot, "")).lower() in ("no", "n", "false", "0"):
                candidates.append((ot, "Yes", "Contrast: turn overtime on"))

            bt = has("BusinessTravel")
            if bt and "frequent" in str(base_features.get(bt, "")).lower():
                candidates.append((bt, "Travel_Rarely", "Reduce travel load"))

            for col_name, hi, hint in (
                ("JobSatisfaction", 4, "Raise job satisfaction to 4"),
                ("EnvironmentSatisfaction", 4, "Raise environment satisfaction to 4"),
                ("RelationshipSatisfaction", 4, "Raise relationship satisfaction to 4"),
                ("WorkLifeBalance", 4, "Raise work–life balance to 4"),
                ("JobInvolvement", 4, "Raise job involvement to 4"),
            ):
                real = has(col_name)
                if not real:
                    continue
                try:
                    cur = int(float(base_features.get(real)))
                except (TypeError, ValueError):
                    continue
                if cur < hi:
                    candidates.append((real, hi, hint))
                elif cur > 1:
                    candidates.append((real, 1, f"Contrast: drop {col_name} to 1"))

            mi = has("MonthlyIncome")
            if mi:
                try:
                    cur = float(base_features.get(mi))
                    candidates.append((mi, round(cur * 1.1, 2), "Try a ~10% pay bump"))
                    candidates.append((mi, round(cur * 0.9, 2), "Contrast: ~10% pay cut"))
                except (TypeError, ValueError):
                    pass

            # Career / mobility signals — often move low-risk cases when satisfaction dials don't
            for col_name, lo_alt, hi_hint, lo_hint in (
                ("NumCompaniesWorked", 0, "More prior employers", "Fewer prior employers"),
                ("YearsSinceLastPromotion", 0, None, "Simulate a recent promotion"),
                ("YearsAtCompany", None, "Longer tenure", "Shorter tenure"),
                ("YearsInCurrentRole", 0, None, "Reset years in role"),
            ):
                real = has(col_name)
                if not real:
                    continue
                try:
                    cur = float(base_features.get(real))
                except (TypeError, ValueError):
                    continue
                if col_name == "NumCompaniesWorked":
                    if cur >= 1:
                        candidates.append((real, 0, lo_hint))
                    candidates.append((real, max(cur + 2, 4), hi_hint))
                elif col_name == "YearsSinceLastPromotion" and cur >= 2:
                    candidates.append((real, 0, lo_hint))
                elif col_name == "YearsAtCompany":
                    candidates.append((real, max(0, cur - 3), lo_hint or "Shorter tenure"))
                    candidates.append((real, cur + 3, hi_hint or "Longer tenure"))
                elif col_name == "YearsInCurrentRole" and cur >= 2:
                    candidates.append((real, 0, lo_hint))

            dist = has("DistanceFromHome")
            if dist:
                try:
                    cur = float(base_features.get(dist))
                    if cur >= 10:
                        candidates.append(
                            (dist, max(1, round(cur * 0.5, 1)), "Shorter commute / hybrid proxy")
                        )
                except (TypeError, ValueError):
                    pass

            ysp = has("YearsSinceLastPromotion")
            if ysp:
                try:
                    cur = int(float(base_features.get(ysp)))
                    if cur >= 2:
                        candidates.append((ysp, 0, "Simulate a recent promotion"))
                except (TypeError, ValueError):
                    pass

            stock = has("StockOptionLevel")
            if stock:
                try:
                    cur = int(float(base_features.get(stock)))
                    if cur <= 0:
                        candidates.append((stock, 1, "Grant stock options (level 1)"))
                except (TypeError, ValueError):
                    pass

            train = has("TrainingTimesLastYear")
            if train:
                try:
                    cur = int(float(base_features.get(train)))
                    candidates.append((train, cur + 1, "Add one training cycle"))
                except (TypeError, ValueError):
                    pass
        else:
            # Telco / generic churn-style dials
            for col_name, alt, hint in (
                ("Contract", "Two year", "Longer contract"),
                ("Contract", "One year", "One-year contract"),
                ("TechSupport", "Yes", "Add tech support"),
                ("OnlineSecurity", "Yes", "Add online security"),
                ("OnlineBackup", "Yes", "Add online backup"),
                ("DeviceProtection", "Yes", "Add device protection"),
                ("PaperlessBilling", "No", "Turn off paperless billing"),
            ):
                real = has(col_name)
                if not real:
                    continue
                cur = base_features.get(real)
                if str(cur) == str(alt):
                    continue
                candidates.append((real, alt, hint))

            for col_name, factor, hint in (
                ("MonthlyCharges", 0.85, "Lower monthly charges ~15%"),
                ("tenure", None, "Add 12 months tenure"),
            ):
                real = has(col_name)
                if not real:
                    continue
                try:
                    cur = float(base_features.get(real))
                except (TypeError, ValueError):
                    continue
                if col_name.lower() == "tenure":
                    candidates.append((real, int(cur) + 12, hint))
                else:
                    candidates.append((real, round(cur * float(factor), 2), hint))

        probes: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for feat, alt, hint in candidates:
            key = f"{feat}|{alt}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            real = feat if feat in feats else colmap.get(str(feat).lower())
            if not real:
                continue
            try:
                pred = self.predict(
                    project_id,
                    _json_safe({**(base_features or {}), real: alt}) or {},
                    include_explanations=False,
                    include_recommendations=False,
                    persist=False,
                    source="simulation",
                )
                after = float(pred.get("probability") or 0)
                delta = after - float(baseline_prob)
            except Exception:
                continue
            probes.append({
                "feature": real,
                "label": real,
                "suggested_value": alt,
                "current": _json_safe((base_features or {}).get(real)),
                "hint": hint,
                "expected_delta": round(delta, 4),
                "expected_probability": round(after, 4),
                "moves_score": abs(delta) >= 0.005,
            })

        # Prefer risk-reducing movers first
        probes.sort(
            key=lambda r: (
                0 if r.get("moves_score") else 1,
                float(r.get("expected_delta") or 0),
            )
        )
        return _json_safe(probes[:12])

    def _probe_driver_nudges(
        self,
        project_id: str,
        base_features: dict[str, Any],
        baseline_prob: float,
        original_pred: dict[str, Any],
        *,
        domain: str,
    ) -> list[dict[str, Any]]:
        """Nudge top risk-raising drivers so focus mode has several editable dials."""
        factors = (
            original_pred.get("explanations", {}).get("drivers")
            or original_pred.get("explanations", {}).get("shap", {}).get("top_features")
            or []
        )
        out: list[dict[str, Any]] = []
        for f in factors[:8]:
            feat = f.get("feature")
            if not feat or feat not in (base_features or {}):
                continue
            impact = float(f.get("impact") or 0)
            if impact <= 0:
                continue
            cur = base_features.get(feat)
            alt = None
            hint = f"Shows up in “why” — test whether changing it moves the score"
            bounds = self._feature_value_bounds(feat, domain)
            try:
                if bounds is not None:
                    lo, hi = bounds
                    v = float(cur)
                    alt = int(hi) if v < hi else int(lo)
            except (TypeError, ValueError):
                alt = None

            if alt is None:
                try:
                    v = float(cur)
                    alt = round(v * 0.85, 2) if abs(v) > 1e-9 else v + 1
                    if abs(float(alt) - v) < 1e-9:
                        alt = v + 1
                    hint = f"Nudge {feat} and re-score"
                except (TypeError, ValueError):
                    s = str(cur).strip().lower()
                    if s in ("yes", "y", "true", "1"):
                        alt = "No"
                    elif s in ("no", "n", "false", "0"):
                        alt = "Yes"
                    else:
                        continue

            try:
                pred = self.predict(
                    project_id,
                    _json_safe({**(base_features or {}), feat: alt}) or {},
                    include_explanations=False,
                    include_recommendations=False,
                    persist=False,
                    source="simulation",
                )
                after = float(pred.get("probability") or 0)
                delta = after - float(baseline_prob)
            except Exception:
                continue
            out.append({
                "feature": feat,
                "label": f.get("label") or feat,
                "suggested_value": alt,
                "current": _json_safe(cur),
                "hint": hint,
                "expected_delta": round(delta, 4),
                "expected_probability": round(after, 4),
                "moves_score": abs(delta) >= 0.005,
            })
        return out

    def _scenario_key_insights(
        self,
        *,
        change_log: list[dict],
        prob_change: float,
        outcome_label: str,
        domain: str,
        moving_levers: Optional[list[dict[str, Any]]] = None,
    ) -> list[str]:
        """Plain bullets describing what the dials mean for this domain."""
        from app.recommendations.domains import DOMAIN_HR_ATTRITION

        insights: list[str] = []
        hr = domain == DOMAIN_HR_ATTRITION
        moved = abs(prob_change) >= 0.005

        if change_log and not moved:
            # Keep this short — plain_summary already states the no-move result
            movers = [
                r for r in (moving_levers or [])
                if abs(float(r.get("expected_delta") or 0)) >= 0.005
            ]
            if movers:
                bits = []
                for r in movers[:2]:
                    label = r.get("label") or r.get("feature")
                    delta = float(r.get("expected_delta") or 0)
                    bits.append(f"{label} → {r.get('suggested_value')} ({delta * 100:+.0f} pts)")
                insights.append("What does move this case: " + "; ".join(bits) + ".")
            else:
                insights.append(
                    "Explanation drivers (like satisfaction) are not always dials that change "
                    "the score — try Show all fields or a higher-risk case."
                )
            return insights

        for c in change_log[:5]:
            feat = str(c.get("feature") or "")
            fl = feat.lower().replace("_", "")
            before, after = c.get("before"), c.get("after")
            label = c.get("label") or feat
            if hr:
                if "overtime" in fl and str(after).lower() in ("no", "false", "0"):
                    insights.append(
                        f"Turning off overtime ({label}) tests workload as a retention lever."
                    )
                elif "businesstravel" in fl:
                    insights.append(
                        f"Lower travel load ({before} → {after}) may ease burnout pressure."
                    )
                elif "jobsatisfaction" in fl or "worklifebalance" in fl:
                    insights.append(
                        f"Improving {label} ({before} → {after}) is a people-ops lever for this case."
                    )
                elif "monthlyincome" in fl:
                    insights.append(
                        f"Higher compensation ({before} → {after}) is a costly but direct stay lever."
                    )
                elif "yearssincelastpromotion" in fl:
                    insights.append(
                        f"Resetting time since promotion ({before} → {after}) proxies a growth-path move."
                    )
                elif "distancefromhome" in fl:
                    insights.append(
                        f"Shorter commute proxy ({before} → {after}) stands in for hybrid flexibility."
                    )
                else:
                    insights.append(f"{label}: {before} → {after}")
            else:
                if "contract" in fl:
                    insights.append(
                        f"Contract shift ({before} → {after}) changes switching friction."
                    )
                elif "charges" in fl or "monthly" in fl:
                    insights.append(
                        f"Price / plan change ({before} → {after}) alters cost pressure."
                    )
                elif "techsupport" in fl or "security" in fl:
                    insights.append(
                        f"Service add-on ({label}: {before} → {after}) can reduce friction."
                    )
                else:
                    insights.append(f"{label}: {before} → {after}")

        ol = outcome_label.lower()
        if hr and "outcome" in ol:
            ol = "attrition"
        if prob_change < -0.05:
            insights.append(
                f"Net effect: chance of {ol} drops by about {abs(prob_change) * 100:.0f} percentage points."
            )
        elif prob_change > 0.05:
            insights.append(
                f"Warning: chance of {ol} rises by about {prob_change * 100:.0f} percentage points."
            )
        elif moved:
            insights.append(
                f"Net effect: chance of {ol} shifts by about {prob_change * 100:+.1f} points."
            )
        return insights

    def _scenario_plain_summary(
        self,
        *,
        outcome_label: str,
        is_regression: bool,
        before: float,
        after: float,
        change: float,
        change_log: list[dict],
        risk_level_change: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> str:
        """One plain-language sentence for the scenario card."""
        from app.recommendations.domains import DOMAIN_HR_ATTRITION

        n = len(change_log)
        change_bits = ", ".join(
            f"{c['label']}: {c['before']} → {c['after']}"
            for c in change_log[:3]
        )
        label = outcome_label
        if domain == DOMAIN_HR_ATTRITION and str(label).lower() in ("outcome", "the outcome", ""):
            label = "attrition"
        elif domain != DOMAIN_HR_ATTRITION and str(label).lower() in ("outcome", "the outcome", ""):
            label = "churn"

        if is_regression:
            direction = "down" if change < 0 else "up" if change > 0 else "flat"
            return (
                f"With {n} tweak(s) ({change_bits or 'no real diffs'}), predicted "
                f"{label} moves from {before:.2f} to {after:.2f} ({direction})."
            )
        pp = change * 100
        if abs(pp) < 0.5:
            return (
                f"With {n} change(s) ({change_bits or 'no real diffs'}), chance of "
                f"{label} stays about {before:.0%} — these dials do not move this person's "
                f"estimate (common on already-low-risk cases)."
            )
        if pp < 0:
            move = f"drops about {abs(pp):.0f} percentage points"
        else:
            move = f"rises about {pp:.0f} percentage points"
        risk_note = ""
        if risk_level_change and risk_level_change != "unchanged":
            risk_note = f" Risk band {risk_level_change}."
        return (
            f"With {n} change(s) ({change_bits or 'no real diffs'}), chance of "
            f"{label} {move} — from {before:.0%} to {after:.0%}.{risk_note}"
        )

    def _is_numeric_feature(self, project: Project, feature: str, value: Any) -> bool:
        """True when a column can take continuous noise for Monte Carlo."""
        cfg = (project.feature_config or {}).get(feature)
        if isinstance(cfg, dict):
            ftype = str(cfg.get("type") or "").lower()
            if ftype in ("categorical", "boolean", "bool", "category"):
                return False
            if ftype in ("numeric", "number", "float", "int", "integer"):
                return True
        if isinstance(value, bool) or value in (True, False):
            return False
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "false", "yes", "no", "y", "n"):
                return False
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _score_feature_dicts(
        self,
        project: Project,
        trained_model,
        model,
        feature_dicts: list[dict[str, Any]],
        *,
        is_regression: bool,
    ) -> np.ndarray:
        """Batch-score raw feature dicts → probabilities or regression values."""
        from app.ml.pipelines.feature_pipeline import FeatureTransformer

        cols = list(project.feature_columns or [])
        raw_df = pd.DataFrame([{c: row.get(c) for c in cols} for row in feature_dicts])
        ft_path = os.path.join(trained_model.model_path, "feature_transformer.joblib")
        if os.path.exists(ft_path):
            transformer = FeatureTransformer()
            transformer.load(trained_model.model_path)
            feature_df = transformer.transform(raw_df)
        else:
            feature_df = raw_df.copy()
            if project.feature_config:
                for col, config in project.feature_config.items():
                    if col.startswith("_") or not isinstance(config, dict):
                        continue
                    if col in feature_df.columns and config.get("type") == "categorical":
                        categories = config.get("categories", [])
                        code_map = config.get("code_map")

                        def _encode(value):
                            if code_map and str(value) in code_map:
                                return code_map[str(value)]
                            if value in categories:
                                return categories.index(value)
                            return -1

                        feature_df[col] = feature_df[col].map(_encode)
            for col in cols:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            if getattr(model, "feature_names", None):
                for col in model.feature_names:
                    if col not in feature_df.columns:
                        feature_df[col] = 0
                feature_df = feature_df[model.feature_names]
            else:
                feature_df = feature_df[[c for c in cols if c in feature_df.columns]]

        if is_regression:
            if hasattr(model, "predict"):
                preds = np.asarray(model.predict(feature_df), dtype=float).reshape(-1)
            else:
                preds = np.asarray(
                    [u.prediction for u in model.predict_with_uncertainty(feature_df)],
                    dtype=float,
                )
            return preds

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feature_df)
            arr = np.asarray(proba, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, 1]
            return arr.reshape(-1)
        return np.asarray(
            [u.prediction for u in model.predict_with_uncertainty(feature_df)],
            dtype=float,
        )

    def _monte_carlo_scenario(
        self,
        project: Project,
        *,
        base_features: dict[str, Any],
        after_features: dict[str, Any],
        locked_features: set[str],
        domain: str,
        n_draws: int,
        noise_scale: float,
        seed: Optional[int],
        is_regression: bool,
    ) -> Optional[dict[str, Any]]:
        """
        Sample feature noise around the case and re-score before/after many times.

        Locked features (user dials) stay fixed so the intervention is held constant.
        Returns percentiles of before/after/delta and P(improve).
        """
        n_draws = int(n_draws) if n_draws is not None else 0
        if n_draws <= 0:
            return None
        n_draws = max(20, min(n_draws, 500))
        # Important: 0.0 is a valid scale (deterministic draws) — do not use `or`
        scale = 0.05 if noise_scale is None else float(noise_scale)
        noise_scale = float(max(0.0, min(scale, 0.25)))

        trained_model = self.get_active_model(project.id)
        if not trained_model:
            return None

        from app.ml.model_loader import load_routed_model

        model = load_routed_model(
            trained_model.model_path,
            problem_type=project.problem_type,
        )

        cols = list(project.feature_columns or [])
        base = {c: (base_features or {}).get(c) for c in cols}
        after = {c: (after_features or {}).get(c) for c in cols}

        numeric_cols: list[str] = []
        for c in cols:
            if c in locked_features:
                continue
            cl = str(c).lower()
            # Skip identifiers — noise on IDs is meaningless and distorts draws
            if cl.endswith("id") or cl.endswith("_id") or cl in {
                "employeenumber",
                "employee_number",
                "customerid",
                "customer_id",
                "rownumber",
                "index",
            }:
                continue
            if self._is_numeric_feature(project, c, base.get(c)):
                numeric_cols.append(c)

        rng = np.random.default_rng(seed if seed is not None else None)
        before_rows: list[dict[str, Any]] = []
        after_rows: list[dict[str, Any]] = []

        for _ in range(n_draws):
            b = dict(base)
            a = dict(after)
            for c in numeric_cols:
                try:
                    x = float(b.get(c))
                except (TypeError, ValueError):
                    continue
                bounds = self._feature_value_bounds(c, domain)
                if bounds is not None:
                    lo, hi = bounds
                    span = max(hi - lo, 1.0)
                    sigma = noise_scale * span
                else:
                    sigma = noise_scale * max(abs(x), 1.0)
                noise = float(rng.normal(0.0, sigma))
                bx = x + noise
                # Mirror the same noise onto the after row so delta isolates the dials
                try:
                    ax = float(a.get(c)) + noise
                except (TypeError, ValueError):
                    ax = bx
                if bounds is not None:
                    lo, hi = bounds
                    bx = min(hi, max(lo, bx))
                    ax = min(hi, max(lo, ax))
                    # keep Likert-ish ints looking like ints
                    if float(lo).is_integer() and float(hi).is_integer() and span <= 10:
                        bx = int(round(bx))
                        ax = int(round(ax))
                b[c] = bx
                a[c] = ax
            before_rows.append(b)
            after_rows.append(a)

        try:
            before_scores = self._score_feature_dicts(
                project, trained_model, model, before_rows, is_regression=is_regression
            )
            after_scores = self._score_feature_dicts(
                project, trained_model, model, after_rows, is_regression=is_regression
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Monte Carlo scoring failed for project %s: %s", project.id, exc
            )
            return None

        before_scores = np.asarray(before_scores, dtype=float).reshape(-1)
        after_scores = np.asarray(after_scores, dtype=float).reshape(-1)
        n = int(min(len(before_scores), len(after_scores)))
        if n < 10:
            return None
        before_scores = before_scores[:n]
        after_scores = after_scores[:n]
        deltas = after_scores - before_scores

        def _pcts(arr: np.ndarray) -> dict[str, float]:
            return {
                "p10": round(float(np.percentile(arr, 10)), 4),
                "p50": round(float(np.percentile(arr, 50)), 4),
                "p90": round(float(np.percentile(arr, 90)), 4),
                "mean": round(float(np.mean(arr)), 4),
            }

        # Align with UI "no change" band (±0.5 pp for classification)
        improve_thresh = -0.005 if not is_regression else -1e-9
        worsen_thresh = 0.005 if not is_regression else 1e-9
        p_improve = float(np.mean(deltas < improve_thresh))
        p_worsen = float(np.mean(deltas > worsen_thresh))
        p_flat = float(np.mean((deltas >= improve_thresh) & (deltas <= worsen_thresh)))

        # Histogram on delta (pp for classification display on FE)
        hist_vals = deltas * 100.0 if not is_regression else deltas
        bins = 11
        counts, edges = np.histogram(hist_vals, bins=bins)
        return {
            "n_draws": n,
            "method": "feature_noise",
            "noise_scale": noise_scale,
            "locked_features": sorted(locked_features),
            "noisy_features": numeric_cols[:40],
            "before": _pcts(before_scores),
            "after": _pcts(after_scores),
            "delta": _pcts(deltas),
            "p_improve": round(p_improve, 3),
            "p_worsen": round(p_worsen, 3),
            "p_unchanged": round(p_flat, 3),
            "histogram": {
                "bin_edges": [round(float(e), 3) for e in edges.tolist()],
                "counts": [int(c) for c in counts.tolist()],
                "unit": "pp" if not is_regression else "value",
            },
            "plain_summary": (
                f"Across {n} noisy draws, chance this change helps is "
                f"{p_improve:.0%} "
                f"(median impact {(float(np.percentile(deltas, 50)) * (100 if not is_regression else 1)):+.1f}"
                f"{' pp' if not is_regression else ''})."
            ),
        }

    def simulate(
        self,
        project_id: str,
        base_features: dict[str, Any],
        modified_features: dict[str, Any],
        *,
        n_draws: int = 200,
        noise_scale: float = 0.05,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Phase 5 what-if: before/after prediction, real feature diffs,
        driver shift, re-ranked recs, and plain-language impact.
        Optional Monte Carlo (n_draws>0) adds outcome distribution under feature noise.
        """
        project = self.get_project(project_id)
        if not project:
            raise ValueError("Project not found")
        is_regression = project.problem_type == "regression"
        outcome_label = str(
            project.target_description or project.target_column or "the outcome"
        )
        from app.recommendations.domains import detect_domain

        domain = detect_domain(
            feature_columns=project.feature_columns,
            features=base_features,
            project_name=project.name,
            target_column=project.target_column,
            target_description=project.target_description,
        )

        # Only real diffs — ignore no-ops
        change_log = self._diff_features(base_features, modified_features or {})
        real_mods = {c["feature"]: (modified_features or {}).get(c["feature"]) for c in change_log}
        real_mods, clamp_notes = self._sanitize_scenario_mods(
            modified=real_mods, domain=domain
        )
        # Drop mods that clamp back to the baseline (e.g. JS 4→10→4)
        effective_mods: dict[str, Any] = {}
        for k, v in real_mods.items():
            before = (base_features or {}).get(k)
            if str(_json_safe(before)) == str(_json_safe(v)):
                continue
            effective_mods[k] = v
        real_mods = effective_mods
        change_log = self._diff_features(base_features, real_mods)
        # If user dialed something but clamp wiped the effective change, keep intent in warnings
        if not change_log and clamp_notes:
            for note in list(clamp_notes):
                if "instead of" in note and "does not change" not in note:
                    clamp_notes.append(
                        "After clamping to the valid range, the value matches the original — "
                        "so the estimate cannot move on this dial alone."
                    )
                    break

        original = self.predict(
            project_id,
            _json_safe(base_features) or {},
            include_explanations=True,
            include_recommendations=False,
            persist=False,
            source="simulation",
        )

        combined = {**(base_features or {}), **real_mods}
        modified = self.predict(
            project_id,
            _json_safe(combined) or {},
            include_explanations=True,
            include_recommendations=True,
            persist=False,
            source="simulation",
        )

        suggested = self._suggested_tweaks(base_features or {}, original, project)

        # User requested a change that clamped to a no-op — still explain honestly
        noop_request = bool(clamp_notes) and not change_log

        mc = self._monte_carlo_scenario(
            project,
            base_features=base_features or {},
            after_features=combined,
            locked_features=set(real_mods.keys()),
            domain=domain,
            n_draws=n_draws,
            noise_scale=noise_scale,
            seed=seed,
            is_regression=is_regression,
        )

        if is_regression:
            original_val = float(original.get("predicted_value") or 0)
            modified_val = float(modified.get("predicted_value") or 0)
            change = modified_val - original_val
            direction = (
                "improved" if change < -1e-9
                else "worsened" if change > 1e-9
                else "unchanged"
            )
            out = {
                "original": {
                    "predicted_value": original_val,
                    "confidence": original.get("confidence", 0),
                    "confidence_interval": original.get("confidence_interval"),
                    "low_confidence": original.get("low_confidence"),
                },
                "modified": {
                    "predicted_value": modified_val,
                    "confidence": modified.get("confidence", 0),
                    "confidence_interval": modified.get("confidence_interval"),
                    "low_confidence": modified.get("low_confidence"),
                },
                "impact": change,
                "impact_percent": round(
                    change * 100 / max(abs(original_val), 0.01), 1
                ) if original_val is not None else 0,
                "direction": direction,
                "modified_features": real_mods,
                "change_log": change_log,
                "warnings": clamp_notes,
                "driver_shift": self._driver_shift(
                    original.get("explanations", {}).get("drivers")
                    or (original.get("explanations") or {}).get("shap", {}).get("top_features"),
                    modified.get("explanations", {}).get("drivers")
                    or (modified.get("explanations") or {}).get("shap", {}).get("top_features"),
                ),
                "plain_summary": self._scenario_plain_summary(
                    outcome_label=outcome_label,
                    is_regression=True,
                    before=original_val,
                    after=modified_val,
                    change=change,
                    change_log=change_log,
                    domain=domain,
                ),
                "suggested_tweaks": suggested,
                "domain": domain,
                "recommendations": modified.get("recommendations") or [],
                "decision_summary": modified.get("decision_summary"),
                "original_explanations": original.get("explanations"),
                "modified_explanations": modified.get("explanations"),
            }
            if mc:
                out["monte_carlo"] = mc
            return out

        original_prob = float(original.get("probability") or 0)
        modified_prob = float(modified.get("probability") or 0)
        prob_change = modified_prob - original_prob
        risk_change = (
            "improved" if prob_change < -0.05
            else "worsened" if prob_change > 0.05
            else "unchanged"
        )

        suggested = self._rank_tweaks_by_impact(
            project_id,
            base_features or {},
            suggested,
            original_prob,
            domain=domain,
        )
        if abs(prob_change) < 0.005 and (change_log or noop_request):
            fallback = self._probe_fallback_levers(
                project_id,
                base_features or {},
                original_prob,
                domain=domain,
                project=project,
            )
            # Prefer probed levers that move the needle
            seen = {s.get("feature") for s in suggested}
            for f in fallback:
                if f.get("feature") not in seen:
                    suggested.append(f)
                    seen.add(f.get("feature"))
            suggested = sorted(
                suggested,
                key=lambda r: float(r.get("expected_delta") or 0),
            )
        # Never surface 0-delta "suggestions" as actionable levers
        suggested = [
            s for s in (suggested or [])
            if abs(float(s.get("expected_delta") or 0)) >= 0.005
        ]

        from app.ml.soft_range import interval_is_soft

        def _soft(pred: dict) -> dict:
            ci = pred.get("confidence_interval") or {}
            return interval_is_soft(
                point=float(pred.get("probability") or 0.5),
                lower=float(ci.get("lower", 0)),
                upper=float(ci.get("upper", 1)),
                low_confidence=bool(pred.get("low_confidence")),
                is_regression=False,
            )

        before_soft = _soft(original)
        after_soft = _soft(modified)

        key_insights = self._scenario_key_insights(
            change_log=change_log,
            prob_change=prob_change,
            outcome_label=outcome_label,
            domain=domain,
            moving_levers=suggested,
        )
        if noop_request and not key_insights:
            key_insights = [
                "That dial is already at (or clamped to) its original value, so the estimate did not move. "
                "Use “Dials that move this case,” or show all fields."
            ]

        plain = self._scenario_plain_summary(
            outcome_label=outcome_label,
            is_regression=False,
            before=original_prob,
            after=modified_prob,
            change=prob_change,
            change_log=change_log or (
                [{"label": "requested dial", "before": "—", "after": "clamped to original"}]
                if noop_request
                else []
            ),
            risk_level_change=risk_change,
            domain=domain,
        )

        out = {
            "original": {
                "probability": original_prob,
                "confidence": original.get("confidence", 0),
                "risk_level": original.get("risk_level", "unknown"),
                "confidence_interval": original.get("confidence_interval"),
                "low_confidence": original.get("low_confidence"),
                "soft_case": bool(before_soft.get("is_soft")),
            },
            "modified": {
                "probability": modified_prob,
                "confidence": modified.get("confidence", 0),
                "risk_level": modified.get("risk_level", "unknown"),
                "confidence_interval": modified.get("confidence_interval"),
                "low_confidence": modified.get("low_confidence"),
                "soft_case": bool(after_soft.get("is_soft")),
            },
            "impact": prob_change,
            "impact_percent": round(
                prob_change * 100 / max(abs(float(original_prob)), 0.01), 1
            ) if original_prob is not None else 0,
            "risk_level_change": risk_change,
            "direction": risk_change,
            "modified_features": real_mods,
            "change_log": change_log,
            "warnings": clamp_notes,
            "driver_shift": self._driver_shift(
                original.get("explanations", {}).get("drivers")
                or (original.get("explanations") or {}).get("shap", {}).get("top_features"),
                modified.get("explanations", {}).get("drivers")
                or (modified.get("explanations") or {}).get("shap", {}).get("top_features"),
            ),
            "plain_summary": plain,
            "key_insights": key_insights,
            "suggested_tweaks": suggested,
            "domain": domain,
            "recommendations": modified.get("recommendations") or [],
            "decision_summary": modified.get("decision_summary"),
            "original_explanations": original.get("explanations"),
            "modified_explanations": modified.get("explanations"),
            "insight_brief_after": modified.get("insight_brief"),
        }
        if mc:
            out["monte_carlo"] = mc
        return out

    def scenario_levers(
        self,
        project_id: str,
        base_features: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Rank what-if dials by *actual* score movement for this case.

        SHAP "drivers" explain contribution; they are not always the dials that
        move the estimate when changed. What-if focus should use this list.
        """
        project = self.get_project(project_id)
        if not project:
            raise ValueError("Project not found")
        if project.problem_type == "regression":
            return {
                "levers": [],
                "plain_summary": "Scenario levers are ranked for yes/no outcomes.",
                "feature_names": [],
            }

        from app.recommendations.domains import detect_domain

        domain = detect_domain(
            feature_columns=project.feature_columns,
            features=base_features,
            project_name=project.name,
            target_column=project.target_column,
            target_description=project.target_description,
        )

        original = self.predict(
            project_id,
            _json_safe(base_features) or {},
            include_explanations=True,
            include_recommendations=False,
            persist=False,
            source="simulation",
        )
        baseline = float(original.get("probability") or 0)
        suggested = self._suggested_tweaks(base_features or {}, original, project)
        ranked = self._rank_tweaks_by_impact(
            project_id,
            base_features or {},
            suggested,
            baseline,
            domain=domain,
        )
        fallback = self._probe_fallback_levers(
            project_id,
            base_features or {},
            baseline,
            domain=domain,
            project=project,
        )
        driver_nudges = self._probe_driver_nudges(
            project_id,
            base_features or {},
            baseline,
            original,
            domain=domain,
        )

        by_feat: dict[str, dict[str, Any]] = {}
        for row in list(ranked or []) + list(fallback or []) + list(driver_nudges or []):
            feat = row.get("feature")
            if not feat:
                continue
            prev = by_feat.get(feat)
            if prev is None or abs(float(row.get("expected_delta") or 0)) > abs(
                float(prev.get("expected_delta") or 0)
            ):
                by_feat[feat] = row

        all_rows = sorted(
            by_feat.values(),
            key=lambda r: (
                0 if abs(float(r.get("expected_delta") or 0)) >= 0.005 else 1,
                float(r.get("expected_delta") or 0),
            ),
        )
        moving = [
            r for r in all_rows if abs(float(r.get("expected_delta") or 0)) >= 0.005
        ]
        # Focus list: movers first, then related explanation fields for editing
        focus_rows: list[dict[str, Any]] = list(moving)
        for r in all_rows:
            if len(focus_rows) >= 6:
                break
            if r.get("feature") in {x.get("feature") for x in focus_rows}:
                continue
            focus_rows.append(r)

        # Sidebar “dials that move” — only real movers (never pad with 0-delta dials)
        display_levers = moving[:8]
        names = [r.get("feature") for r in focus_rows if r.get("feature")]
        n_move = len(moving)
        if n_move >= 2:
            plain = (
                f"{n_move} dials change this person's estimate. "
                f"Focus shows {len(names)} related fields to edit."
            )
        elif n_move == 1:
            only = display_levers[0].get("label") or display_levers[0].get("feature")
            plain = (
                f"Only “{only}” strongly moves this estimate among the dials we probed. "
                f"Satisfaction / overtime may not shift a calm case like this."
            )
        else:
            plain = (
                "None of the usual dials move this estimate much — the score is already flat "
                "for this person. Try Show all fields, or pick a higher-risk case."
            )

        return {
            "levers": _json_safe(display_levers),
            "feature_names": names,
            "baseline_probability": baseline,
            "plain_summary": plain,
            "domain": domain,
            "moving_count": n_move,
        }
    
    # =========================================================================
    # Feedback (A7: outcome log — not decision ledger B3)
    # =========================================================================

    def record_feedback(
        self,
        prediction_id: str,
        actual_outcome: str,
        action_taken: Optional[str] = None,
        *,
        project_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Record (or update) real-world outcome for a project prediction.

        Outcomes (classification):
          - positive / yes / 1 / true  → matched to project target_positive_label
          - negative / no / 0 / false
          - unknown
        Or free text that matches target labels.

        Returns structured match vs model call + feedback payload.
        notes is accepted for API compatibility but not persisted (no column yet).
        """
        del notes  # reserved for B3 ledger; not stored on ProjectPrediction
        prediction = self.db.query(ProjectPrediction).filter(
            ProjectPrediction.id == prediction_id
        ).first()
        if not prediction:
            return None

        project = self.get_project(prediction.project_id)
        if not project:
            return None
        if project_id and prediction.project_id != project_id:
            return None

        is_regression = project.problem_type == "regression"
        outcome_norm = self._normalize_outcome(
            actual_outcome, project, is_regression=is_regression
        )
        if outcome_norm is None:
            raise ValueError(
                f"Invalid actual_outcome '{actual_outcome}'. "
                "Use positive/yes, negative/no, unknown, or a label matching your target."
            )

        action_code = (action_taken or "").strip() or None
        prediction.actual_outcome = outcome_norm["stored"]
        prediction.action_taken = action_code
        prediction.feedback_date = datetime.utcnow()
        self.db.commit()
        self.db.refresh(prediction)
        self._invalidate_effectiveness_cache(prediction.project_id)

        return self._format_feedback_record(prediction, project, outcome_norm)

    def _normalize_outcome(
        self,
        raw: str,
        project: Project,
        *,
        is_regression: bool,
    ) -> Optional[dict[str, Any]]:
        s = str(raw).strip()
        if not s:
            return None
        low = s.lower()
        pos_label = str(project.target_positive_label or "1")

        if is_regression:
            try:
                val = float(s)
            except ValueError:
                return None
            return {
                "stored": str(val),
                "binary": None,
                "kind": "numeric",
                "display": s,
            }

        if low in ("unknown", "unk", "n/a", "na", "pending"):
            return {
                "stored": "unknown",
                "binary": None,
                "kind": "unknown",
                "display": "Unknown",
            }

        positive_aliases = {
            "positive",
            "yes",
            "true",
            "1",
            "y",
            "churned",
            "attrited",
            "attrition",
            "left",
            "resigned",
            "quit",
            "terminated",
            pos_label.lower(),
        }
        negative_aliases = {
            "negative",
            "no",
            "false",
            "0",
            "n",
            "retained",
            "not_churned",
            "stayed",
            "active",
            "employed",
        }

        if low in positive_aliases or s == pos_label:
            display = "Yes" if pos_label in ("1", "Yes", "yes", "True") else pos_label
            return {
                "stored": "positive",
                "binary": 1,
                "kind": "known",
                "display": display,
            }
        if low in negative_aliases:
            return {
                "stored": "negative",
                "binary": 0,
                "kind": "known",
                "display": "No",
            }
        return None

    def _format_feedback_record(
        self,
        prediction: ProjectPrediction,
        project: Project,
        outcome_norm: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        is_regression = project.problem_type == "regression"
        if outcome_norm is None and prediction.actual_outcome:
            outcome_norm = self._normalize_outcome(
                prediction.actual_outcome, project, is_regression=is_regression
            ) or {
                "stored": prediction.actual_outcome,
                "binary": None,
                "kind": "raw",
                "display": prediction.actual_outcome,
            }

        predicted_positive = None
        if not is_regression and prediction.probability is not None:
            predicted_positive = float(prediction.probability) >= get_decision_threshold(project)

        actual_binary = outcome_norm.get("binary") if outcome_norm else None
        match = None
        if predicted_positive is not None and actual_binary is not None:
            match = bool(predicted_positive) == bool(actual_binary)

        outcome_label = project.target_description or project.target_column or "outcome"
        plain = "Outcome logged."
        if match is True:
            plain = (
                f"Model Yes/No call agreed with the real {outcome_label} "
                f"(predicted {float(prediction.probability or 0):.0%})."
            )
        elif match is False:
            plain = (
                f"Model Yes/No call disagreed with the real {outcome_label} "
                f"(predicted {float(prediction.probability or 0):.0%}). "
                "Use this miss when reviewing drivers."
            )
        elif outcome_norm and outcome_norm.get("kind") == "unknown":
            plain = "Marked unknown — no accuracy credit either way."

        return {
            "prediction_id": prediction.id,
            "project_id": prediction.project_id,
            "actual_outcome": prediction.actual_outcome,
            "actual_display": (outcome_norm or {}).get("display") or prediction.actual_outcome,
            "action_taken": prediction.action_taken,
            "feedback_date": (
                prediction.feedback_date.isoformat() if prediction.feedback_date else None
            ),
            "predicted_probability": prediction.probability,
            "predicted_value": prediction.predicted_value,
            "risk_level": prediction.risk_level,
            "predicted_positive": predicted_positive,
            "model_matched_outcome": match,
            "plain_summary": plain,
            "notes_stored": False,
        }

    def get_prediction_case(self, project_id: str, prediction_id: str) -> dict[str, Any]:
        """Return a stored prediction shaped for the case brief / deep-links."""
        project = self.get_project(project_id)
        if not project:
            raise ValueError("Project not found")
        prediction = (
            self.db.query(ProjectPrediction)
            .filter(
                ProjectPrediction.id == prediction_id,
                ProjectPrediction.project_id == project_id,
            )
            .first()
        )
        if not prediction:
            raise ValueError("Prediction not found")

        is_regression = project.problem_type == "regression"
        outcome_label = project.target_description or project.target_column or "outcome"
        drivers = list(prediction.top_factors or [])
        feedback = None
        if prediction.actual_outcome:
            feedback = self._format_feedback_record(prediction, project)

        if is_regression and prediction.predicted_value is not None:
            plain = (
                f"Stored case: predicted {outcome_label} ≈ "
                f"{float(prediction.predicted_value):.4g}."
            )
        elif prediction.probability is not None:
            plain = (
                f"Stored case: about {float(prediction.probability):.0%} chance of "
                f"{outcome_label}."
            )
        else:
            plain = "Stored case opened from Priorities."

        blindspots = None
        blindspot_warnings = []
        annotated_drivers = drivers
        if drivers and not is_regression:
            from app.ml.blindspot import annotate_drivers, detect_blindspots

            # Deep-link hydrate: heuristics only (no parquet reload — keep Priorities snappy)
            blindspots = detect_blindspots(
                top_factors=drivers,
                features=prediction.features or {},
                feature_config=project.feature_config,
                consistency=None,
                training_data=None,
                target_column=project.target_column,
                target_positive_label=project.target_positive_label,
                outcome_label=str(outcome_label),
            )
            annotated_drivers = annotate_drivers(drivers, blindspots)
            blindspot_warnings = blindspots.get("warnings") or []

        # Restore conformal interval / abstention so TrustSpine matches Don't-act flags
        trust_meta = {}
        shap_blob = prediction.shap_values
        if isinstance(shap_blob, dict) and isinstance(shap_blob.get("_knowa_trust"), dict):
            trust_meta = shap_blob.get("_knowa_trust") or {}

        confidence_interval = trust_meta.get("confidence_interval")
        abstention_reason = trust_meta.get("abstention_reason")
        model_disagreement = trust_meta.get("model_disagreement")
        low_confidence = bool(prediction.low_confidence) or bool(
            trust_meta.get("low_confidence")
        )

        if confidence_interval is None and prediction.features:
            try:
                live = self.predict(
                    project_id,
                    prediction.features or {},
                    entity_id=prediction.entity_id,
                    include_explanations=False,
                    include_recommendations=False,
                    persist=False,
                    source="hydrate",
                )
                confidence_interval = live.get("confidence_interval")
                abstention_reason = abstention_reason or live.get("abstention_reason")
                model_disagreement = (
                    model_disagreement
                    if model_disagreement is not None
                    else live.get("model_disagreement")
                )
                low_confidence = low_confidence or bool(live.get("low_confidence"))
            except Exception:
                confidence_interval = None

        if confidence_interval is None and not is_regression and prediction.probability is not None:
            # Last resort: wide band so Don't-act cases don't show a fake tight spine
            from app.ml.soft_range import NEAR_FULL_WIDTH

            p = float(prediction.probability)
            if low_confidence:
                confidence_interval = {
                    "lower": 0.0,
                    "upper": min(1.0, max(NEAR_FULL_WIDTH, p + 0.35)),
                    "level": 0.9,
                    "width": min(1.0, max(NEAR_FULL_WIDTH, p + 0.35)),
                }

        soft_reason = None
        if not is_regression and prediction.probability is not None and confidence_interval:
            from app.ml.soft_range import interval_is_soft

            soft = interval_is_soft(
                point=float(prediction.probability),
                lower=float(confidence_interval.get("lower", 0)),
                upper=float(confidence_interval.get("upper", 1)),
                low_confidence=low_confidence,
                is_regression=False,
            )
            soft_reason = soft.get("reason")
            if soft.get("is_soft"):
                low_confidence = True

        return {
            "prediction_id": prediction.id,
            "entity_id": prediction.entity_id,
            "probability": prediction.probability,
            "predicted_value": prediction.predicted_value,
            "confidence": prediction.confidence,
            "risk_level": prediction.risk_level,
            "problem_type": project.problem_type,
            "target": outcome_label,
            "features": prediction.features or {},
            "low_confidence": low_confidence,
            "confidence_interval": confidence_interval,
            "abstention_reason": abstention_reason,
            "model_disagreement": model_disagreement,
            "soft_reason": soft_reason,
            "explanations": {
                "drivers": annotated_drivers,
                "shap": {"top_features": annotated_drivers},
            },
            "recommendations": prediction.recommendations or [],
            "feedback": feedback,
            "blindspots": blindspots,
            "blindspot_warnings": blindspot_warnings,
            "plain_summary": plain,
            "source": "history",
            "persisted": True,
            "created_at": (
                prediction.created_at.isoformat() if prediction.created_at else None
            ),
        }

    def get_feedback(
        self, project_id: str, prediction_id: str
    ) -> Optional[dict[str, Any]]:
        project = self.get_project(project_id)
        if not project:
            return None
        prediction = self.db.query(ProjectPrediction).filter(
            ProjectPrediction.id == prediction_id,
            ProjectPrediction.project_id == project_id,
        ).first()
        if not prediction or not prediction.actual_outcome:
            return None
        return self._format_feedback_record(prediction, project)

    def _invalidate_effectiveness_cache(self, project_id: str) -> None:
        _EFFECTIVENESS_CACHE.pop(f"{self.org_id}:{project_id}", None)

    def _effectiveness_fingerprint(self, project_id: str) -> int:
        """Cheap version stamp so cache refreshes when outcomes change."""
        n_pred = (
            self.db.query(ProjectPrediction)
            .filter(
                ProjectPrediction.project_id == project_id,
                ProjectPrediction.action_taken.isnot(None),
                ProjectPrediction.actual_outcome.isnot(None),
            )
            .count()
        )
        n_dec = (
            self.db.query(Decision)
            .filter(
                Decision.project_id == project_id,
                Decision.organization_id == self.org_id,
                Decision.actual_outcome.isnot(None),
                Decision.action_code.isnot(None),
            )
            .count()
        )
        return int(n_pred) * 1_000_003 + int(n_dec)

    def get_action_effectiveness(self, project_id: str) -> dict[str, dict]:
        """
        Map action_code -> {n, success_rate, success_n, action_name} for A5 blend.

        For churn/attrition-style positives: success = outcome negative (avoided
        the bad event) — retained customer / employee stayed.

        Sources: ProjectPrediction feedback + Decision ledger outcomes (deduped).
        Cached briefly on the predict hot path.
        """
        from app.recommendations.action_catalog import get_action
        from app.recommendations.domains import detect_domain

        project = self.get_project(project_id)
        if not project:
            return {}

        cache_key = f"{self.org_id}:{project_id}"
        fingerprint = self._effectiveness_fingerprint(project_id)
        cached = _EFFECTIVENESS_CACHE.get(cache_key)
        if (
            cached
            and cached[1] == fingerprint
            and (time.time() - cached[0]) < _EFFECTIVENESS_TTL_SEC
        ):
            return cached[2]

        domain = detect_domain(
            feature_columns=project.feature_columns,
            project_name=project.name,
            target_column=project.target_column,
            target_description=project.target_description,
        )
        is_regression = project.problem_type == "regression"
        rows = (
            self.db.query(
                ProjectPrediction.id,
                ProjectPrediction.action_taken,
                ProjectPrediction.actual_outcome,
            )
            .filter(
                ProjectPrediction.project_id == project_id,
                ProjectPrediction.action_taken.isnot(None),
                ProjectPrediction.actual_outcome.isnot(None),
            )
            .all()
        )
        stats: dict[str, dict[str, float]] = {}
        seen_pred_ids: set[str] = set()
        for p in rows:
            code = (p.action_taken or "").strip()
            if not code:
                continue
            seen_pred_ids.add(p.id)
            norm = self._normalize_outcome(
                p.actual_outcome, project, is_regression=is_regression
            )
            if not norm or norm.get("binary") is None:
                continue
            bucket = stats.setdefault(code, {"n": 0, "success_n": 0})
            bucket["n"] += 1
            # Success = avoided positive outcome (retained / stayed)
            if int(norm["binary"]) == 0:
                bucket["success_n"] += 1

        # Include ledger outcomes that never made it onto a prediction row
        dec_rows = (
            self.db.query(
                Decision.id,
                Decision.prediction_id,
                Decision.action_code,
                Decision.actual_outcome,
            )
            .filter(
                Decision.project_id == project_id,
                Decision.organization_id == self.org_id,
                Decision.action_code.isnot(None),
                Decision.actual_outcome.isnot(None),
                Decision.status.in_(["committed", "checking", "closed"]),
            )
            .all()
        )
        for d in dec_rows:
            if d.prediction_id and d.prediction_id in seen_pred_ids:
                continue
            code = (d.action_code or "").strip()
            if not code:
                continue
            norm = self._normalize_outcome(
                d.actual_outcome, project, is_regression=is_regression
            )
            if not norm or norm.get("binary") is None:
                continue
            bucket = stats.setdefault(code, {"n": 0, "success_n": 0})
            bucket["n"] += 1
            if int(norm["binary"]) == 0:
                bucket["success_n"] += 1

        out = {}
        for code, b in stats.items():
            n = int(b["n"])
            if n < 1:
                continue
            action = get_action(code, domain=domain)
            success_n = int(b["success_n"])
            rate = round(success_n / n, 4)
            reliable = n >= 3
            if not reliable:
                note = (
                    f"Logged {success_n}/{n} favorable — need 3+ before rankings shift."
                )
            elif rate >= 0.65:
                note = f"Favorable in {success_n}/{n} cases — ranking slightly boosted."
            elif rate <= 0.35:
                note = f"Only {success_n}/{n} went well — ranking tempered."
            else:
                note = f"Mixed ({success_n}/{n} favorable) — mild ranking adjustment."
            out[code] = {
                "n": n,
                "n_outcomes": n,
                "success_n": success_n,
                "success_rate": rate,
                "effectiveness_rate": rate,
                "action_name": action.name if action else code,
                "domain": domain,
                "reliable": reliable,
                "learning_note": note,
            }
        _EFFECTIVENESS_CACHE[cache_key] = (time.time(), fingerprint, out)
        return out

    def get_feedback_summary(
        self, project_id: str, limit: int = 200
    ) -> dict[str, Any]:
        """A7 loop dashboard: coverage, accuracy, action effectiveness."""
        from app.recommendations.domains import DOMAIN_HR_ATTRITION, detect_domain

        project = self.get_project(project_id)
        if not project:
            raise ValueError("Project not found")

        domain = detect_domain(
            feature_columns=project.feature_columns,
            project_name=project.name,
            target_column=project.target_column,
            target_description=project.target_description,
        )
        hr = domain == DOMAIN_HR_ATTRITION
        bad_event = "attrition" if hr else "churn"
        good_event = "stayed" if hr else "retained"

        is_regression = project.problem_type == "regression"
        total_preds = (
            self.db.query(ProjectPrediction)
            .filter(ProjectPrediction.project_id == project_id)
            .count()
        )
        n_fb = (
            self.db.query(ProjectPrediction)
            .filter(
                ProjectPrediction.project_id == project_id,
                ProjectPrediction.actual_outcome.isnot(None),
            )
            .count()
        )
        # Match / recent sample over a bounded window; coverage uses full COUNT above
        with_fb = (
            self.db.query(ProjectPrediction)
            .filter(
                ProjectPrediction.project_id == project_id,
                ProjectPrediction.actual_outcome.isnot(None),
            )
            .order_by(ProjectPrediction.feedback_date.desc())
            .limit(limit)
            .all()
        )
        outcomes: dict[str, int] = {}
        agree = 0
        known = 0
        recent = []
        for p in with_fb:
            key = str(p.actual_outcome)
            outcomes[key] = outcomes.get(key, 0) + 1
            rec = self._format_feedback_record(p, project)
            if rec.get("model_matched_outcome") is not None:
                known += 1
                if rec["model_matched_outcome"]:
                    agree += 1
            recent.append({
                "prediction_id": p.id,
                "actual_outcome": p.actual_outcome,
                "action_taken": p.action_taken,
                "probability": p.probability,
                "matched": rec.get("model_matched_outcome"),
                "feedback_date": rec.get("feedback_date"),
            })

        action_eff = self.get_action_effectiveness(project_id)
        acc = (agree / known) if known else None
        coverage = (n_fb / total_preds) if total_preds else 0.0

        ranked_actions = sorted(
            (
                {
                    "action_code": code,
                    **meta,
                }
                for code, meta in action_eff.items()
            ),
            key=lambda x: (x.get("success_rate", 0), x.get("n", 0)),
            reverse=True,
        )

        if n_fb == 0:
            plain = (
                f"No real outcomes logged yet. After a case, record whether the person "
                f"{'left or stayed' if hr else 'churned or was retained'} "
                f"(or Unknown), and optionally which action you took — "
                f"so the loop can score model hits and action effectiveness."
            )
        else:
            plain = (
                f"{n_fb} of {total_preds} predictions have outcomes logged "
                f"({coverage:.0%} coverage). "
            )
            if acc is not None:
                plain += (
                    f"Among known Yes/No labels, model matched {agree}/{known} ({acc:.0%}). "
                )
            if ranked_actions:
                best = ranked_actions[0]
                name = best.get("action_name") or best.get("action_code")
                plain += (
                    f"Strongest logged action so far: {name} "
                    f"({best['success_n']}/{best['n']} {good_event} — avoided {bad_event})"
                )
                if best.get("reliable"):
                    plain += " — enough sample to gently reshape recommendation rankings."
                else:
                    plain += " — still a small sample (<3); rankings stay on the catalog."
                plain += ". "
            plain += "Outcomes feed A7 learning; follow-up autopsy lives on the decision ledger."

        reliable_n = sum(1 for a in ranked_actions if a.get("reliable"))
        return {
            "project_id": project_id,
            "problem_type": project.problem_type,
            "domain": domain,
            "supported": not is_regression,
            "total_predictions": total_preds,
            "with_feedback": n_fb,
            "coverage_rate": round(coverage, 4),
            "outcome_distribution": outcomes,
            "known_outcome_count": known,
            "model_match_count": agree,
            "model_match_rate": round(acc, 4) if acc is not None else None,
            "action_effectiveness": action_eff,
            "action_effectiveness_ranked": ranked_actions,
            "learning": {
                "min_n": 3,
                "actions_with_outcomes": len(ranked_actions),
                "actions_reshaping_rankings": reliable_n,
                "plain": (
                    f"{reliable_n} action(s) have enough outcomes to reshape rankings."
                    if reliable_n
                    else "No action yet has 3+ outcomes — recommendations still use the catalog."
                ),
            },
            "recent": recent[:20],
            "plain_summary": plain,
            "layer": "A7_outcome_learning",
            "not_included": [],
            "included_layers": [
                "A7_outcome_learning",
                "B3_scheduled_rechecks",
                "B4_causal_blindspots",
            ],
        }
