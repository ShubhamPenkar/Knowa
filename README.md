# Explainable Business Decision Intelligence Platform

A closed-loop decision intelligence system that predicts business outcomes, explains predictions using Explainable AI (SHAP/LIME), generates actionable insights, recommends optimal actions, supports what-if simulation, and learns from feedback.

## 🎯 Key Features

### 1. Prediction Layer
- Multiple ML models: XGBoost, LightGBM, Random Forest, Logistic Regression
- Stacked ensemble (meta-learner on out-of-fold base predictions)
- Conformal prediction intervals + low_confidence abstention on disagreement / wide CI

### 2. Explainability Layer
- **SHAP** explanations (global + local)
- **LIME** explanations (local approximation)
- **Explanation Consistency Score** - comparing SHAP vs LIME for trust assessment

### 3. Insight Generation
- Transforms technical feature importance into business-friendly language
- Rule templates + NLP mapping
- Severity classification (critical, warning, info, positive)

### 4. Decision Recommendation
- Hybrid scoring system (Impact × Cost × Relevance)
- 15+ pre-configured business actions
- Ranked recommendations with reasoning

### 5. What-If Simulation
- Modify feature values interactively
- See before/after prediction comparison
- Get actionable recommendations based on changes

### 6. Feedback Learning Loop
- Track action outcomes
- Update model performance metrics
- Automatic retraining triggers

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- (Optional) Docker
- macOS + XGBoost: `brew install libomp` if you hit OpenMP/`libomp.dylib` errors

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Train initial model
python scripts/train_models.py --generate-data

# Seed database
python scripts/seed_data.py

# Start server
uvicorn app.main:app --reload
```

Backend runs at: http://localhost:8000

API docs: http://localhost:8000/docs

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend runs at: http://localhost:3000

### Docker Setup

```bash
docker-compose up --build
```

## 📁 Project Structure

```
Knowa/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration
│   │   ├── database.py          # DB connection
│   │   ├── routes/              # API endpoints
│   │   ├── services/            # Business logic
│   │   ├── ml/
│   │   │   ├── models/          # ML model implementations
│   │   │   ├── explainers/      # SHAP, LIME, consistency
│   │   │   └── pipelines/       # Training & prediction
│   │   ├── insights/            # Insight generation
│   │   ├── recommendations/     # Action scoring
│   │   ├── schemas/             # Pydantic models
│   │   └── db/                  # SQLAlchemy models
│   ├── scripts/                 # CLI scripts
│   └── data/                    # Data & models
│
├── frontend/
│   ├── src/
│   │   ├── pages/               # React pages
│   │   ├── components/          # UI components
│   │   └── services/            # API client
│   └── package.json
│
└── docker-compose.yml
```

## 📊 API Endpoints

All routes are under `/api` (not `/api/v1`).

### SaaS

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/signup` | POST | Create org + owner |
| `/api/auth/login` | POST | JWT login |
| `/api/datasets` | POST | Upload CSV |
| `/api/projects` | POST | Create project |
| `/api/projects/{id}/train` | POST | Train model |
| `/api/projects/{id}/predict` | POST | Predict (incl. CI / low_confidence) |

### Demo (fixed churn schema)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Create churn prediction |
| `/api/explain/{id}` | GET | Get SHAP/LIME explanations |
| `/api/insights/{id}` | GET | Get business insights |
| `/api/recommend/{id}` | GET | Get ranked actions |
| `/api/simulate` | POST | Run what-if simulation |
| `/api/feedback` | POST | Submit outcome feedback |
| `/api/model/train` | POST | Trigger model training |
| `/api/model/metrics` | GET | Get performance metrics |

**First run:** from `backend/`, train before using demo predict:

`python scripts/train_models.py --generate-data`

## 🔧 Configuration

Key settings in `backend/app/config.py`:

```python
# Model settings
default_model = "xgboost"  # or ensemble
confidence_threshold = 0.7
explanation_consistency_threshold = 0.7

# Conformal / abstention (Phase 1a)
conformal_alpha = 0.1
disagreement_threshold = 0.25
interval_width_threshold = 0.85
stacking_n_folds = 5

# Model routing (Phase 1b)
routing_mode = "auto"  # auto | foundation_model | ensemble
foundation_max_rows = 10_000
foundation_max_features = 500
prefer_tabpfn = True  # needs optional `pip install tabpfn`

# Phase 1.5 quality training
test_size = 0.2
calib_size = 0.2
enable_optuna = True
optuna_trials = 12
probability_calibration = "isotonic"  # isotonic | sigmoid | none
early_stopping_rounds = 40

# Recommendation weights
impact_weight = 0.5
cost_weight = 0.3
relevance_weight = 0.2

# Feedback settings
feedback_window_days = 90
retrain_threshold_samples = 100
```

## 📈 Evaluation Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Model | Accuracy, Precision, Recall, F1, AUC-ROC | Standard ML metrics |
| Explainability | Consistency Score | SHAP vs LIME agreement (0-1) |
| Explainability | Trust Level | high/medium/low based on consistency |
| Decision | Action Effectiveness | Retention rate when action taken |
| Decision | Decision Score | Weighted impact × cost × relevance |

## 🔄 Workflow

```
Data Input → Prediction → Explanation → Insight Generation →
Action Recommendation → What-if Simulation →
Action Taken → Feedback → Model Improvement
```

## 🛠️ Development

### Running Tests
```bash
cd backend
pytest tests/
```

### Training Models
```bash
# Train ensemble (all models)
python scripts/train_models.py --model-type ensemble

# Train specific model
python scripts/train_models.py --model-type xgboost

# Generate synthetic data
python scripts/train_models.py --generate-data --n-samples 10000
```

## 📝 Sample Request

```python
import requests

# Create prediction
response = requests.post("http://localhost:8000/api/predict", json={
    "features": {
        "tenure": 12,
        "monthly_charges": 65.5,
        "contract_type": "month-to-month",
        "payment_method": "electronic_check",
        "internet_service": "fiber_optic",
        "online_security": "no",
        "tech_support": "no",
        "streaming_tv": "yes",
        "streaming_movies": "yes",
        "num_support_tickets": 3,
        "days_since_last_interaction": 45,
        "num_complaints": 1,
        "satisfaction_score": 3.2,
        "total_charges": 786.0
    }
})

prediction = response.json()
print(f"Churn Probability: {prediction['churn_probability']}")
print(f"Risk Level: {prediction['churn_risk_level']}")
print(f"Confidence: {prediction['confidence_score']}")
```

## 🎓 For Research/Academic Use

This project demonstrates:
- Multi-model ensemble with uncertainty quantification
- SHAP vs LIME explanation consistency analysis
- Automated insight generation from ML explanations
- Decision-theoretic action recommendation
- Closed-loop learning from outcomes

Suitable for:
- Final year engineering projects
- ML/AI research papers
- Industry proof-of-concepts

## 📄 License

MIT License
