# CLAUDE.md

## 🧠 Project Name

Explainable Business Decision Intelligence Platform

---

## 🎯 Project Vision

This project is a **Closed-Loop Explainable Decision Intelligence System** that enables businesses to:

* Predict outcomes (e.g., customer churn, sales trends)
* Understand *why* predictions occur (Explainable AI)
* Generate business-friendly insights automatically
* Recommend optimal actions (not just rules, but scored decisions)
* Simulate “what-if” scenarios before taking action
* Learn from real-world outcomes to continuously improve

This is NOT just a machine learning model.
This is a **full decision intelligence system with feedback learning**.

---

## 🚀 Core Innovation

Unlike traditional systems that stop at prediction or visualization, this platform:

1. Bridges **prediction → explanation → action**
2. Introduces **Explanation Consistency Scoring (SHAP vs LIME)**
3. Converts feature importance into **human-readable business insights**
4. Uses **decision scoring instead of static recommendations**
5. Supports **scenario simulation (what-if analysis)**
6. Implements a **feedback loop for continuous learning**

---

## 🏗️ System Architecture Overview

### 1. Prediction Layer

Responsible for predicting outcomes like churn.

Models:

* XGBoost / LightGBM (primary)
* Logistic Regression (baseline + calibration)
* Random Forest (robustness check)

Enhancements:

* Output prediction probability
* Add **confidence score** using model agreement / probability distribution

---

### 2. Explainability Layer

Explains why a prediction was made.

Tools:

* SHAP → global + local explanations
* LIME → local explanation cross-validation

Enhancement:

* **Explanation Consistency Score**

  * Compare SHAP vs LIME outputs
  * If mismatch → flag prediction as low trust

---

### 3. Insight Generation Layer (Critical)

Transforms technical outputs into business insights.

Input:

* SHAP feature importance values

Output:

* Human-readable insights

Example:

* Input: low_app_usage = -0.42
* Output: “Customer engagement is low, increasing churn risk”

Implementation:

* Rule templates + NLP mapping
* No vague outputs — must be actionable and clear

---

### 4. Decision Intelligence Layer

Recommends business actions.

Approach:

* Hybrid system:

  * Rule-based engine
  * Scoring-based optimization

Each action must include:

* Impact Score (effect on prediction)
* Cost Score
* Final Decision Score

Example:

* Offer discount → high retention impact, medium cost
* Send notification → low cost, moderate impact

System should rank actions, not just list them.

---

### 5. What-if Simulation Engine

Allows users to test decisions before applying them.

Functionality:

* Modify input features (e.g., login frequency, complaints)
* Re-run prediction + explanation
* Show before vs after comparison

Output Example:

* Before: churn risk = 82%
* After: churn risk = 45%
* Impact: -37%

---

### 6. Feedback Learning System (Critical for novelty)

System must learn from real outcomes.

Process:

* Track actions taken
* Track actual outcomes
* Update:

  * Model performance
  * Recommendation effectiveness

Goal:
Create a **self-improving decision system**

---

## 🔄 End-to-End Workflow

Data → Prediction → Explanation → Insight Generation →
Decision Recommendation → What-if Simulation →
Action Taken → Feedback → Model Improvement

---

## ⚙️ Tech Stack Guidelines

### Backend

* Python
* FastAPI (preferred)
* Scikit-learn, XGBoost, LightGBM
* SHAP, LIME

### Frontend

* React.js
* Clean, modern UI (dashboard style)

### Database

* PostgreSQL or MongoDB

---

## 📁 Suggested Folder Structure

backend/

* app.py
* routes/
* models/
* services/

  * prediction_service.py
  * explainability_service.py
  * insight_service.py
  * recommendation_service.py
  * simulation_service.py
  * feedback_service.py

frontend/

* src/

  * components/
  * pages/
  * services/

---

## 📊 Evaluation Metrics

### Model Performance

* Accuracy
* Precision, Recall, F1-score

### Explainability

* Explanation Consistency Score (SHAP vs LIME similarity)

### Decision Intelligence

* Action Effectiveness Score (before vs after outcome improvement)

---

## ❗ Important Development Rules

* DO NOT use pre-trained models
* Train models from scratch using dataset
* Code must be modular and scalable
* Avoid hardcoding logic — keep systems extensible
* Every prediction must be explainable
* Every explanation must translate into an insight
* Every insight must map to an action

---

## 🧠 How Claude Should Behave

Claude must act as:

* Senior ML Engineer
* Backend Architect
* Product-aware Developer

Claude should:

* Think in systems, not isolated code
* Always justify design decisions
* Prefer modular, production-ready code
* Avoid shortcuts or vague implementations
* Clearly separate each layer of architecture
* Suggest improvements if architecture can be optimized

---

## 📌 Development Strategy

DO NOT build everything at once.

Follow this order:

1. Prediction Layer
2. Explainability Layer
3. Insight Generation
4. Recommendation Engine
5. Simulation Engine
6. Feedback System
7. Frontend integration

---

## 🧩 Future Extensions (Optional)

* Reinforcement learning for decision optimization
* AutoML for model tuning
* Real-time streaming data
* Industry-specific customization (finance, e-commerce, SaaS)

---

## 🛑 What This Project Is NOT

* Not just a dashboard
* Not just a churn model
* Not just SHAP visualization

It is a:
👉 **Decision Intelligence Platform with Explainability and Learning Loop**

---

## ✅ Expected Output Quality

All outputs must be:

* Structured
* Explainable
* Actionable
* Business-friendly

Avoid:

* Raw technical dumps
* Uninterpreted feature values
* Black-box predictions

---

## 🧾 Final Note

This project should be treated as:

* A final-year engineering project
* A research-level system
* A potential startup-grade product

Focus on **clarity, modularity, and innovation**.

