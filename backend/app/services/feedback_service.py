"""Feedback service for outcome tracking and learning loop."""

from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Prediction, Feedback, ActionCatalog, ActionEffectiveness
from app.schemas import FeedbackCreate

settings = get_settings()


class FeedbackService:
    """
    Manages feedback loop:
    1. Record outcomes
    2. Calculate effectiveness
    3. Trigger retraining when needed
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def submit_feedback(self, feedback_data: FeedbackCreate) -> dict[str, Any]:
        """
        Submit outcome feedback.
        
        Args:
            feedback_data: Feedback submission data
            
        Returns:
            Stored feedback record
        """
        # Validate prediction exists
        prediction = self.db.query(Prediction).filter(
            Prediction.id == feedback_data.prediction_id
        ).first()
        
        if not prediction:
            raise ValueError(f"Prediction {feedback_data.prediction_id} not found")
        
        # Check for existing feedback
        existing = self.db.query(Feedback).filter(
            Feedback.prediction_id == feedback_data.prediction_id
        ).first()
        
        if existing:
            raise ValueError(f"Feedback already exists for prediction {feedback_data.prediction_id}")
        
        # Get action ID if provided
        action_id = None
        if feedback_data.action_taken_code:
            action = self.db.query(ActionCatalog).filter(
                ActionCatalog.action_code == feedback_data.action_taken_code
            ).first()
            if action:
                action_id = action.id
        
        # Create feedback record
        feedback = Feedback(
            prediction_id=feedback_data.prediction_id,
            action_taken_id=action_id,
            actual_outcome=feedback_data.actual_outcome,
            outcome_date=feedback_data.outcome_date,
            notes=feedback_data.notes,
        )
        self.db.add(feedback)
        
        # Update action effectiveness if action was taken
        if action_id and feedback_data.actual_outcome in ["churned", "retained"]:
            self._update_action_effectiveness(
                action_id,
                was_successful=(feedback_data.actual_outcome == "retained"),
                prediction=prediction
            )
        
        self.db.commit()
        self.db.refresh(feedback)
        
        # Check if retraining is needed
        self._check_retrain_trigger()
        
        return self._format_feedback(feedback)
    
    def get_feedback(self, prediction_id: str) -> Optional[dict[str, Any]]:
        """Get feedback for prediction."""
        feedback = self.db.query(Feedback).filter(
            Feedback.prediction_id == prediction_id
        ).first()
        
        if not feedback:
            return None
        
        return self._format_feedback(feedback)
    
    def get_summary_stats(self, days: int = 30) -> dict[str, Any]:
        """Get feedback summary statistics."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Get all feedback in period
        feedback_records = (
            self.db.query(Feedback)
            .filter(Feedback.recorded_at >= cutoff)
            .all()
        )
        
        total = len(feedback_records)
        if total == 0:
            return {
                "period_days": days,
                "total_feedback": 0,
                "outcome_distribution": {},
                "model_accuracy": None,
                "action_effectiveness": {},
            }
        
        # Outcome distribution
        outcomes = {}
        for fb in feedback_records:
            outcomes[fb.actual_outcome] = outcomes.get(fb.actual_outcome, 0) + 1
        
        # Calculate model accuracy
        correct = 0
        total_known = 0
        for fb in feedback_records:
            if fb.actual_outcome in ["churned", "retained"]:
                total_known += 1
                prediction = self.db.query(Prediction).filter(
                    Prediction.id == fb.prediction_id
                ).first()
                if prediction:
                    predicted_churn = prediction.churn_probability >= 0.5
                    actual_churn = fb.actual_outcome == "churned"
                    if predicted_churn == actual_churn:
                        correct += 1
        
        accuracy = correct / total_known if total_known > 0 else None
        
        # Action effectiveness
        action_stats = self._get_action_effectiveness_stats()
        
        return {
            "period_days": days,
            "total_feedback": total,
            "outcome_distribution": outcomes,
            "model_accuracy": round(accuracy, 4) if accuracy else None,
            "action_effectiveness": action_stats,
            "retrain_recommended": self._should_retrain(accuracy),
        }
    
    def _update_action_effectiveness(
        self,
        action_id: str,
        was_successful: bool,
        prediction: Prediction
    ) -> None:
        """Update action effectiveness metrics."""
        # Get or create current period effectiveness record
        today = datetime.utcnow().date()
        period_start = today.replace(day=1)  # First of month
        
        if today.month == 12:
            period_end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            period_end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
        
        effectiveness = (
            self.db.query(ActionEffectiveness)
            .filter(
                ActionEffectiveness.action_id == action_id,
                ActionEffectiveness.period_start == period_start
            )
            .first()
        )
        
        if not effectiveness:
            effectiveness = ActionEffectiveness(
                action_id=action_id,
                period_start=period_start,
                period_end=period_end,
                times_recommended=0,
                times_taken=0,
                success_rate=0.0,
                avg_probability_reduction=0.0,
            )
            self.db.add(effectiveness)
        
        # Update counts
        effectiveness.times_taken = (effectiveness.times_taken or 0) + 1
        
        # Update success rate (exponential moving average)
        n = effectiveness.times_taken
        alpha = 1 / n if n < 100 else 0.01
        old_rate = effectiveness.success_rate or 0.5
        effectiveness.success_rate = (1 - alpha) * old_rate + alpha * (1 if was_successful else 0)
    
    def _get_action_effectiveness_stats(self) -> dict[str, Any]:
        """Get effectiveness stats for all actions."""
        # Get recent effectiveness records
        cutoff = datetime.utcnow() - timedelta(days=90)
        
        stats = {}
        actions = self.db.query(ActionCatalog).filter(ActionCatalog.is_active == True).all()
        
        for action in actions:
            effectiveness = (
                self.db.query(ActionEffectiveness)
                .filter(
                    ActionEffectiveness.action_id == action.id,
                    ActionEffectiveness.period_start >= cutoff
                )
                .all()
            )
            
            if effectiveness:
                total_taken = sum(e.times_taken or 0 for e in effectiveness)
                avg_success = sum((e.success_rate or 0) * (e.times_taken or 0) for e in effectiveness)
                if total_taken > 0:
                    avg_success /= total_taken
                
                stats[action.action_code] = {
                    "times_taken": total_taken,
                    "success_rate": round(avg_success, 4),
                }
        
        return stats
    
    def _check_retrain_trigger(self) -> bool:
        """Check if model retraining should be triggered."""
        # Count recent feedback
        cutoff = datetime.utcnow() - timedelta(days=settings.feedback_window_days)
        
        recent_count = (
            self.db.query(func.count(Feedback.id))
            .filter(Feedback.recorded_at >= cutoff)
            .scalar()
        )
        
        if recent_count >= settings.retrain_threshold_samples:
            # Could trigger async retraining job here
            return True
        
        return False
    
    def _should_retrain(self, accuracy: Optional[float]) -> bool:
        """Determine if retraining is recommended."""
        if accuracy is None:
            return False
        
        # Check if accuracy dropped significantly
        # In production, compare against baseline
        return accuracy < (1 - settings.accuracy_drop_threshold)
    
    def _format_feedback(self, feedback: Feedback) -> dict[str, Any]:
        """Format feedback for response."""
        action_code = None
        if feedback.action_taken_id:
            action = self.db.query(ActionCatalog).filter(
                ActionCatalog.id == feedback.action_taken_id
            ).first()
            if action:
                action_code = action.action_code
        
        return {
            "id": feedback.id,
            "prediction_id": feedback.prediction_id,
            "action_taken": action_code,
            "actual_outcome": feedback.actual_outcome,
            "outcome_date": feedback.outcome_date,
            "recorded_at": feedback.recorded_at,
        }
