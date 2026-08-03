"""Project management service for SaaS."""

import os
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Project, Dataset, TrainedModel, CustomAction, ProjectPrediction
from app.services.dataset_service import DatasetService

settings = get_settings()


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
        """Create new prediction project."""
        # Validate dataset
        dataset = self.db.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.organization_id == self.org_id
        ).first()
        if not dataset:
            raise ValueError("Dataset not found")
        
        # Validate columns exist
        column_names = [col["name"] for col in dataset.columns]
        if target_column not in column_names:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        invalid_features = [f for f in feature_columns if f not in column_names]
        if invalid_features:
            raise ValueError(f"Feature columns not found: {invalid_features}")
        
        if target_column in feature_columns:
            raise ValueError("Target column cannot be in feature columns")
        
        # Create project
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
        from app.ml.models import EnsembleModel
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
            
            # Prepare features and target
            X = df[project.feature_columns].copy()
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
                    raise ValueError(
                        f"Target column '{project.target_column}' must contain at least two classes. "
                        f"Found: {unique_values}"
                    )
                positive_label = str(project.target_positive_label)
                if positive_label not in unique_values:
                    normalized = {val.lower(): val for val in unique_values}
                    candidate_keys = [
                        "1", "yes", "true", "y", "positive", "churned", "churn"
                    ]
                    matches = [normalized[key] for key in candidate_keys if key in normalized]
                    if len(matches) == 1:
                        positive_label = matches[0]
                        project.target_positive_label = positive_label
                        self.db.commit()
                    else:
                        raise ValueError(
                            f"Target positive label '{positive_label}' not found in column "
                            f"'{project.target_column}'. Found values: {unique_values}"
                        )
                y = (target_values == positive_label).astype(int)
                class_counts = y.value_counts()
                if class_counts.min() < 2:
                    raise ValueError(
                        "Training requires at least 2 samples in each class. "
                        f"Class distribution: {class_counts.to_dict()}"
                    )
            
            # Handle categorical columns FIRST (before any numeric operations)
            feature_config = {}
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category' or X[col].dtype.name == 'str':
                    # Encode categorical
                    categories = [str(c) for c in X[col].dropna().unique().tolist()]
                    X[col] = X[col].astype('category').cat.codes
                    feature_config[col] = {"type": "categorical", "categories": categories}
                else:
                    # Fill numeric nulls with median
                    if X[col].isnull().any():
                        X[col] = X[col].fillna(X[col].median())
                    feature_config[col] = {"type": "numeric"}
            
            # Save feature config
            project.feature_config = feature_config
            
            # Train/Test Split (80/20)
            stratify_labels = y if not is_regression else None
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                X, y, df.index, test_size=0.2, random_state=42, stratify=stratify_labels
            )
            if not is_regression:
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    raise ValueError(
                        "Train/test split resulted in a single-class set. "
                        "Please use a dataset with more balanced classes."
                    )
            
            # Train model on training data only
            start_time = time.time()
            model = EnsembleModel(problem_type=project.problem_type)
            metrics = model.train(X_train, y_train, validation_data=(X_test, y_test))
            training_time = time.time() - start_time
            
            # Save model
            version = f"v{len(project.trained_models) + 1}"
            model_path = os.path.join(self.model_dir, project.id, version)
            os.makedirs(model_path, exist_ok=True)
            model.save(model_path)
            
            # Save training data for SHAP
            X_train.to_parquet(os.path.join(model_path, "training_data.parquet"))
            
            # Save test set with original values for prediction UI
            test_df = df.iloc[test_idx].copy()
            test_df.to_parquet(os.path.join(model_path, "test_data.parquet"))
            
            # Calculate feature importance
            try:
                explainer = SHAPExplainer(model.get_primary_model(), background_data=X_train)
                global_result = explainer.explain_global(X_train)
                importance = global_result.get("feature_importance", {})
            except:
                importance = {}
            
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
                    training_samples=len(X),
                    training_time_seconds=training_time,
                    is_active=True,
                )
            else:
                trained_model = TrainedModel(
                    project_id=project_id,
                    version=version,
                    model_path=model_path,
                    # Classification metrics
                    accuracy=metrics.get("ensemble_accuracy"),
                    precision_score=metrics.get("ensemble_precision"),
                    recall_score=metrics.get("ensemble_recall"),
                    f1_score=metrics.get("ensemble_f1_score"),
                    auc_roc=metrics.get("ensemble_auc_roc"),
                    feature_importance=importance,
                    training_samples=len(X),
                    training_time_seconds=training_time,
                    is_active=True,
                )
            
            self.db.add(trained_model)
            
            project.status = "trained"
            self.db.commit()
            self.db.refresh(trained_model)
            
            return trained_model
            
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
    ) -> dict[str, Any]:
        """Make prediction for a project."""
        from app.ml.models import EnsembleModel
        from app.ml.explainers.shap_explainer import SHAPExplainer
        
        project = self.get_project(project_id)
        if not project or project.status not in ["trained", "ready"]:
            raise ValueError("Project not found or not trained")
        
        trained_model = self.get_active_model(project_id)
        if not trained_model:
            raise ValueError("No trained model available")
        
        is_regression = project.problem_type == "regression"
        
        # Load model
        model = EnsembleModel(problem_type=project.problem_type)
        model.load(trained_model.model_path)
        
        # Prepare features
        feature_df = pd.DataFrame([features])
        
        # Apply feature encoding
        if project.feature_config:
            for col, config in project.feature_config.items():
                if col in feature_df.columns:
                    if config["type"] == "categorical":
                        categories = config.get("categories", [])
                        value = feature_df[col].iloc[0]
                        if value in categories:
                            feature_df[col] = categories.index(value)
                        else:
                            feature_df[col] = -1  # Unknown category
        
        # Ensure all feature columns present
        for col in project.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0  # Default value
        
        feature_df = feature_df[project.feature_columns]
        
        # Make prediction (different for regression vs classification)
        if is_regression:
            predicted_value = float(model.predict(feature_df)[0])
            confidence = float(model.get_confidence(feature_df)[0])
            
            result = {
                "predicted_value": predicted_value,
                "confidence": confidence,
                "target": project.target_description,
                "problem_type": "regression",
            }
        else:
            probability = float(model.predict_proba(feature_df)[0])
            confidence = float(model.get_confidence(feature_df)[0])
            risk_level = self._get_risk_level(probability)
            
            result = {
                "probability": probability,
                "confidence": confidence,
                "risk_level": risk_level,
                "target": project.target_description,
                "problem_type": "classification",
            }
        
        # Get explanations
        shap_values = None
        top_factors = None
        if include_explanations:
            try:
                training_data = pd.read_parquet(
                    os.path.join(trained_model.model_path, "training_data.parquet")
                )
                # Get the primary model from ensemble for SHAP
                primary_model = model.get_primary_model()
                explainer = SHAPExplainer(primary_model, background_data=training_data)
                shap_result = explainer.explain_instance(feature_df)
                
                top_factors = []
                for feat in shap_result["explanations"][:5]:
                    top_factors.append({
                        "feature": feat["feature"],
                        "value": features.get(feat["feature"]),
                        "impact": feat["shap_value"],
                        "direction": "increases" if feat["shap_value"] > 0 else "decreases",
                    })
                
                result["explanations"] = {
                    "shap": {
                        "top_features": top_factors,
                        "base_value": shap_result["base_value"],
                    },
                    "all_factors": shap_result["explanations"],
                }
            except Exception as e:
                result["explanations"] = {"error": str(e)}
        
        # Generate business insights from SHAP
        insights = []
        if top_factors and include_explanations:
            insights = self._generate_business_insights(top_factors, features, project)
            result["insights"] = insights
        
        # Get recommendations (only for classification)
        recommendations = None
        if include_recommendations and not is_regression:
            recommendations = self._get_recommendations(project_id, result.get("probability", 0.5), features, top_factors)
            result["recommendations"] = recommendations
        
        # Store prediction
        if is_regression:
            prediction = ProjectPrediction(
                project_id=project_id,
                model_version=trained_model.version,
                entity_id=entity_id,
                features=features,
                predicted_value=predicted_value,
                confidence=confidence,
                shap_values=shap_values,
                top_factors=top_factors,
            )
        else:
            prediction = ProjectPrediction(
                project_id=project_id,
                model_version=trained_model.version,
                entity_id=entity_id,
                features=features,
                probability=result["probability"],
                confidence=confidence,
                risk_level=result["risk_level"],
                shap_values=shap_values,
                top_factors=top_factors,
                recommendations=recommendations,
            )
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)
        
        result["prediction_id"] = prediction.id
        return result
    
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
        top_factors: Optional[list]
    ) -> list[dict]:
        """Get action recommendations based on prediction."""
        # Get org's custom actions
        project = self.get_project(project_id)
        actions = self.db.query(CustomAction).filter(
            CustomAction.organization_id == self.org_id,
            CustomAction.is_active == True
        ).all()
        
        if not actions:
            return []
        
        recommendations = []
        for action in actions:
            # Calculate relevance based on applicable conditions
            relevance = 0.5
            if action.applicable_when and top_factors:
                for factor in top_factors:
                    if factor["feature"] in str(action.applicable_when):
                        relevance = 0.8
                        break
            
            # Score calculation
            impact_score = action.estimated_impact * probability
            cost_normalized = min(action.estimated_cost / 1000, 1)  # Normalize cost
            final_score = 0.5 * impact_score + 0.3 * (1 - cost_normalized) + 0.2 * relevance
            
            recommendations.append({
                "action_code": action.code,
                "action_name": action.name,
                "description": action.description,
                "impact_score": round(impact_score, 3),
                "cost": action.estimated_cost,
                "final_score": round(final_score, 3),
                "reasoning": self._generate_reasoning(action, probability, top_factors),
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x["final_score"], reverse=True)
        for i, rec in enumerate(recommendations):
            rec["rank"] = i + 1
        
        return recommendations[:5]  # Top 5
    
    def _generate_reasoning(self, action: CustomAction, probability: float, top_factors: Optional[list]) -> str:
        """Generate human-readable reasoning for recommendation."""
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
        project: Project
    ) -> list[dict]:
        """Generate business-friendly insights from SHAP values with actionable suggestions."""
        insights = []
        target = project.target_description or project.target_column
        is_regression = project.problem_type == "regression"
        
        # Feature-specific insight templates with reasons and suggestions
        feature_insights = {
            # Customer engagement features
            "logins": {
                "high_impact_positive": {
                    "reason": "Low login frequency indicates disengagement from your platform",
                    "suggestion": "Implement re-engagement campaigns: personalized emails, push notifications about new features, or exclusive content to bring the user back"
                },
                "high_impact_negative": {
                    "reason": "Regular login activity shows this customer is actively engaged",
                    "suggestion": "Maintain engagement by offering loyalty rewards or early access to new features"
                }
            },
            "spend": {
                "high_impact_positive": {
                    "reason": "Low spending pattern suggests declining interest or perceived value",
                    "suggestion": "Offer targeted discounts, bundle deals, or personalized product recommendations to increase purchase value"
                },
                "high_impact_negative": {
                    "reason": "Consistent spending indicates strong customer value perception",
                    "suggestion": "Upsell premium features or introduce a loyalty program to maximize customer lifetime value"
                }
            },
            "tenure": {
                "high_impact_positive": {
                    "reason": "Short tenure customers are still in the evaluation phase and more likely to leave",
                    "suggestion": "Focus on onboarding experience, provide dedicated support, and showcase value early in the customer journey"
                },
                "high_impact_negative": {
                    "reason": "Long-term customers have established habits and are less likely to switch",
                    "suggestion": "Reward loyalty with exclusive benefits and ask for referrals"
                }
            },
            "balance": {
                "high_impact_positive": {
                    "reason": "Low account balance may indicate financial constraints or shifting priorities",
                    "suggestion": "Offer flexible payment plans, highlight cost-saving features, or provide budget-friendly alternatives"
                },
                "high_impact_negative": {
                    "reason": "Healthy account balance shows financial commitment to your service",
                    "suggestion": "Introduce premium tier options or investment opportunities"
                }
            },
            "tickets": {
                "high_impact_positive": {
                    "reason": "High support ticket volume indicates frustration or unresolved issues",
                    "suggestion": "Prioritize resolving their issues, offer proactive support outreach, and consider assigning a dedicated account manager"
                },
                "high_impact_negative": {
                    "reason": "Few support issues suggest a smooth user experience",
                    "suggestion": "Collect positive feedback and testimonials, offer to be a case study"
                }
            },
            "isactivemember": {
                "high_impact_positive": {
                    "reason": "Inactive membership status strongly correlates with customer departure",
                    "suggestion": "Reach out with win-back offers, highlight unused benefits, or conduct an exit interview to understand barriers"
                },
                "high_impact_negative": {
                    "reason": "Active membership shows ongoing engagement with your service",
                    "suggestion": "Recognize their activity with milestone rewards or exclusive member benefits"
                }
            },
            "numofproducts": {
                "high_impact_positive": {
                    "reason": "Single product customers have weaker platform attachment",
                    "suggestion": "Cross-sell complementary products, offer bundle discounts, or demonstrate how additional products solve their problems"
                },
                "high_impact_negative": {
                    "reason": "Multi-product customers are deeply integrated and have higher switching costs",
                    "suggestion": "Continue deepening the relationship with personalized product recommendations"
                }
            },
            "age": {
                "high_impact_positive": {
                    "reason": "Age demographics may indicate different expectations or needs",
                    "suggestion": "Customize communication style and channel preferences based on demographic patterns"
                },
                "high_impact_negative": {
                    "reason": "This age group tends to show strong retention patterns",
                    "suggestion": "Understand what resonates with this demographic and apply to similar customers"
                }
            },
            "creditscore": {
                "high_impact_positive": {
                    "reason": "Credit score patterns may indicate financial stress affecting purchase decisions",
                    "suggestion": "Offer flexible payment options, highlight value for money, or provide cost-reduction tips"
                },
                "high_impact_negative": {
                    "reason": "Strong financial standing correlates with stable customer relationships",
                    "suggestion": "Offer premium services or investment-related features"
                }
            },
            "geography": {
                "high_impact_positive": {
                    "reason": "Geographic location may correlate with regional competition or service gaps",
                    "suggestion": "Investigate regional competitors, consider location-specific promotions or improved local support"
                },
                "high_impact_negative": {
                    "reason": "This region shows strong customer retention",
                    "suggestion": "Study success factors in this region and replicate in other areas"
                }
            },
            "estimatedsalary": {
                "high_impact_positive": {
                    "reason": "Income level may affect price sensitivity and perceived value",
                    "suggestion": "Ensure pricing aligns with perceived value, offer tiered options, or emphasize ROI"
                },
                "high_impact_negative": {
                    "reason": "Income bracket shows good fit with your pricing model",
                    "suggestion": "Consider premium upsell opportunities"
                }
            }
        }
        
        for factor in top_factors[:5]:
            feature = factor.get("feature", "Unknown")
            impact = factor.get("impact", 0)
            value = factor.get("value", features.get(feature, "N/A"))
            feature_lower = feature.lower().replace("_", "").replace(" ", "")
            
            # Determine impact direction
            is_positive_impact = impact > 0  # Pushing towards target (e.g., churn)
            impact_strength = abs(impact)
            
            # Get feature-specific insights if available
            feature_template = feature_insights.get(feature_lower, None)
            
            if feature_template:
                key = "high_impact_positive" if is_positive_impact else "high_impact_negative"
                template = feature_template.get(key, {})
                reason = template.get("reason", f"{feature.replace('_', ' ').title()} is affecting the prediction")
                suggestion = template.get("suggestion", "Monitor this factor closely")
            else:
                # Generic but still informative fallback
                if is_positive_impact:
                    reason = f"The current value of {feature.replace('_', ' ').title()} ({value}) is increasing {target} risk"
                    suggestion = f"Consider strategies to improve {feature.replace('_', ' ').lower()} metrics"
                else:
                    reason = f"The {feature.replace('_', ' ').title()} value ({value}) is favorable and reducing {target} risk"
                    suggestion = f"Maintain current {feature.replace('_', ' ').lower()} levels"
            
            # Determine severity based on impact strength
            if impact_strength > 0.15:
                severity = "critical" if is_positive_impact else "positive"
            elif impact_strength > 0.05:
                severity = "warning" if is_positive_impact else "info"
            else:
                severity = "info"
            
            insights.append({
                "feature": feature,
                "value": value,
                "impact": impact,
                "impact_strength": "high" if impact_strength > 0.15 else ("medium" if impact_strength > 0.05 else "low"),
                "direction": "increasing" if is_positive_impact else "decreasing",
                "severity": severity,
                "reason": reason,
                "suggestion": suggestion,
                "message": f"{reason}. {suggestion}"
            })
        
        # Sort by absolute impact (most important first)
        insights.sort(key=lambda x: abs(x["impact"]), reverse=True)
        
        return insights
    
    # =========================================================================
    # Simulation
    # =========================================================================
    
    def simulate(
        self,
        project_id: str,
        base_features: dict[str, Any],
        modified_features: dict[str, Any],
    ) -> dict[str, Any]:
        """Simulate what-if scenario."""
        project = self.get_project(project_id)
        is_regression = project.problem_type == "regression"
        
        # Get original prediction
        original = self.predict(
            project_id, base_features,
            include_explanations=True,
            include_recommendations=False
        )
        
        # Merge features
        combined = {**base_features, **modified_features}
        
        # Get modified prediction
        modified = self.predict(
            project_id, combined,
            include_explanations=True,
            include_recommendations=True
        )
        
        # Calculate changes based on problem type
        if is_regression:
            original_val = original.get("predicted_value", 0)
            modified_val = modified.get("predicted_value", 0)
            change = modified_val - original_val
            
            return {
                "original": {
                    "predicted_value": original_val,
                    "confidence": original.get("confidence", 0),
                },
                "modified": {
                    "predicted_value": modified_val,
                    "confidence": modified.get("confidence", 0),
                },
                "impact": change,
                "impact_percent": round(change * 100 / max(abs(original_val), 0.01), 1) if original_val else 0,
                "modified_features": modified_features,
                "recommendations": modified.get("recommendations", []),
            }
        else:
            original_prob = original.get("probability", 0)
            modified_prob = modified.get("probability", 0)
            prob_change = modified_prob - original_prob
            
            return {
                "original": {
                    "probability": original_prob,
                    "confidence": original.get("confidence", 0),
                    "risk_level": original.get("risk_level", "unknown"),
                },
                "modified": {
                    "probability": modified_prob,
                    "confidence": modified.get("confidence", 0),
                    "risk_level": modified.get("risk_level", "unknown"),
                },
                "impact": prob_change,
                "impact_percent": round(prob_change * 100 / max(original_prob, 0.01), 1) if original_prob else 0,
                "risk_level_change": "improved" if prob_change < -0.05 else "worsened" if prob_change > 0.05 else "unchanged",
                "modified_features": modified_features,
                "recommendations": modified.get("recommendations", []),
            }
    
    # =========================================================================
    # Feedback
    # =========================================================================
    
    def record_feedback(
        self,
        prediction_id: str,
        actual_outcome: str,
        action_taken: Optional[str] = None,
    ) -> bool:
        """Record outcome feedback for a prediction."""
        prediction = self.db.query(ProjectPrediction).filter(
            ProjectPrediction.id == prediction_id
        ).first()
        
        if not prediction:
            return False
        
        # Verify project belongs to org
        project = self.get_project(prediction.project_id)
        if not project:
            return False
        
        prediction.actual_outcome = actual_outcome
        prediction.action_taken = action_taken
        prediction.feedback_date = datetime.utcnow()
        self.db.commit()
        
        return True
