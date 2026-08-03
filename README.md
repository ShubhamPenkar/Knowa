# Explainable Business Decision Intelligence Platform

A closed-loop decision intelligence system that predicts business outcomes, explains predictions using Explainable AI (SHAP/LIME), generates actionable insights, recommends optimal actions, supports what-if simulation, and learns from feedback.

## 🎯 Key Features

### 1. Prediction Layer
- Multiple ML models: XGBoost, LightGBM, Random Forest, Logistic Regression
- Ensemble model for robust predictions
- Confidence scoring based on model agreement

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
xai/
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

## 🔧 Configuration

Key settings in `backend/app/config.py`:

```python
# Model settings
default_model = "xgboost"  # or ensemble
confidence_threshold = 0.7
explanation_consistency_threshold = 0.7

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
