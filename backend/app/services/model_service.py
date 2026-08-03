"""Model management service."""

import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import ModelPerformance, ActionCatalog, ActionEffectiveness
from app.ml.models import get_model, EnsembleModel
from app.ml.pipelines.training_pipeline import TrainingPipeline

settings = get_settings()


class ModelService:
    """
    Manages model training, metrics, and status.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def train_model(self, model_type: str = "ensemble") -> dict[str, Any]:
        """
        Train a model from scratch.
        
        Args:
            model_type: Type of model to train
            
        Returns:
            Training results and metrics
        """
        pipeline = TrainingPipeline(
            model_type=model_type,
            model_path=settings.model_path
        )
        
        # Load data
        data_path = os.path.join("data", "processed", "churn_data.parquet")
        if os.path.exists(data_path):
            df = pd.read_parquet(data_path)
        else:
            # Use sample data generator
            from app.ml.pipelines.preprocessing import generate_sample_data
            df = generate_sample_data(n_samples=5000)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_parquet(data_path)
        
        # Train
        results = pipeline.train(df, target_column="churn")
        
        # Store metrics
        self._store_metrics(model_type, results["metrics"])
        
        return results
    
    def get_metrics(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent model performance metrics."""
        metrics = (
            self.db.query(ModelPerformance)
            .order_by(ModelPerformance.evaluation_date.desc())
            .limit(limit)
            .all()
        )
        
        return [self._format_metrics(m) for m in metrics]
    
    def get_version_metrics(self, model_version: str) -> Optional[dict[str, Any]]:
        """Get metrics for specific model version."""
        metrics = (
            self.db.query(ModelPerformance)
            .filter(ModelPerformance.model_version == model_version)
            .order_by(ModelPerformance.evaluation_date.desc())
            .first()
        )
        
        if not metrics:
            return None
        
        return self._format_metrics(metrics)
    
    def get_action_effectiveness(self) -> list[dict[str, Any]]:
        """Get effectiveness stats for all actions."""
        actions = self.db.query(ActionCatalog).filter(ActionCatalog.is_active == True).all()
        
        result = []
        for action in actions:
            # Get aggregate effectiveness
            effectiveness = (
                self.db.query(ActionEffectiveness)
                .filter(ActionEffectiveness.action_id == action.id)
                .all()
            )
            
            total_recommended = sum(e.times_recommended or 0 for e in effectiveness)
            total_taken = sum(e.times_taken or 0 for e in effectiveness)
            
            if total_taken > 0:
                weighted_success = sum(
                    (e.success_rate or 0) * (e.times_taken or 0)
                    for e in effectiveness
                )
                avg_success = weighted_success / total_taken
                
                weighted_reduction = sum(
                    (e.avg_probability_reduction or 0) * (e.times_taken or 0)
                    for e in effectiveness
                )
                avg_reduction = weighted_reduction / total_taken
            else:
                avg_success = 0
                avg_reduction = 0
            
            result.append({
                "action_code": action.action_code,
                "action_name": action.action_name,
                "times_recommended": total_recommended,
                "times_taken": total_taken,
                "adoption_rate": total_taken / total_recommended if total_recommended > 0 else 0,
                "success_rate": round(avg_success, 4),
                "avg_probability_reduction": round(avg_reduction, 4),
            })
        
        return result
    
    def get_status(self) -> dict[str, Any]:
        """Get current model status."""
        model_path = os.path.join(settings.model_path, "ensemble")
        single_model_path = os.path.join(settings.model_path, f"{settings.default_model}.joblib")
        
        model_loaded = False
        model_version = None
        model_type = None
        
        if os.path.exists(model_path):
            try:
                model = EnsembleModel()
                model.load(model_path)
                model_loaded = True
                model_version = model.version
                model_type = "ensemble"
            except Exception:
                pass
        elif os.path.exists(single_model_path):
            try:
                model = get_model(settings.default_model)
                model.load(single_model_path)
                model_loaded = True
                model_version = model.version
                model_type = settings.default_model
            except Exception:
                pass
        
        # Get latest metrics
        latest_metrics = (
            self.db.query(ModelPerformance)
            .order_by(ModelPerformance.evaluation_date.desc())
            .first()
        )
        
        return {
            "model_loaded": model_loaded,
            "model_type": model_type,
            "model_version": model_version,
            "model_path": settings.model_path,
            "latest_metrics": self._format_metrics(latest_metrics) if latest_metrics else None,
            "default_model": settings.default_model,
        }
    
    def _store_metrics(self, model_type: str, metrics: dict[str, float]) -> None:
        """Store training metrics."""
        version = f"{model_type}_v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        performance = ModelPerformance(
            model_version=version,
            accuracy=metrics.get("accuracy"),
            precision_score=metrics.get("precision"),
            recall_score=metrics.get("recall"),
            f1_score=metrics.get("f1_score"),
            auc_roc=metrics.get("auc_roc"),
            explanation_consistency_avg=metrics.get("explanation_consistency_avg"),
            sample_size=metrics.get("sample_size"),
        )
        self.db.add(performance)
        self.db.commit()
    
    def _format_metrics(self, metrics: ModelPerformance) -> dict[str, Any]:
        """Format metrics for response."""
        return {
            "model_version": metrics.model_version,
            "evaluation_date": metrics.evaluation_date,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision_score,
            "recall": metrics.recall_score,
            "f1_score": metrics.f1_score,
            "auc_roc": metrics.auc_roc,
            "explanation_consistency_avg": metrics.explanation_consistency_avg,
            "sample_size": metrics.sample_size,
        }
