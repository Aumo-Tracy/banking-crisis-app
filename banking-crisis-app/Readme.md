# 🏦 African Banking Crisis Predictor
### Africa AI Hub — AISIP Cohort 1 Capstone Project | Pathway 4: AI Engineering

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://eapetn5usvi494bqhm3kkn.streamlit.app/)

> **Live App:** https://eapetn5usvi494bqhm3kkn.streamlit.app/

---

## 📌 Problem Statement

Banking crises in Africa have historically devastated economies —
wiping out household savings, collapsing small businesses, and
reversing decades of development. Ghana (2017), Zimbabwe (2000s),
and Nigeria (2009) are recent examples.

This project builds a **machine learning early warning system**
that predicts banking crisis risk from economic indicators —
giving policymakers, central banks, and financial institutions
a data-driven tool to intervene before collapse occurs.

---

## 🎯 Solution

An end-to-end ML pipeline: from raw historical data to a
**deployed web application** that accepts economic inputs
and returns a crisis risk prediction in real time.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | Kaggle — Africa Economic Banking and Systemic Crisis Data |
| Records | 1,059 rows |
| Countries | 13 African countries |
| Time Period | 1860 – 2014 |
| Target | Banking Crisis (binary: crisis / no crisis) |

---

## 🔧 ML Pipeline

### Steps Taken
1. **Data Loading & EDA** — shape, info, describe, 5 visualisations
2. **Cleaning** — dropped identifiers, handled nulls, encoded target
3. **Feature Engineering** — StandardScaler, SMOTE for class imbalance
4. **Model Training** — 3 models trained and compared
5. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix
6. **Deployment** — Streamlit web app with live URL

---

## 🤖 Models Compared

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.958% | 0.750% | 0.789% | 0.769% |
| Random Forest ⭐ | 0.976% | 0.938% | 0.780% | 0.857% |
| XGBoost | 0.967% | 0.833% | 0.789% | 0.811% |

> ⭐ **Best Model: Random Forest** — selected for deployment

---

## 🌐 Deployed App

**Live URL:** https://eapetn5usvi494bqhm3kkn.streamlit.app/

The app allows users to input 9 economic indicators and
receive an instant banking crisis risk prediction with
probability score.

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualisation |
| Scikit-learn | ML models and evaluation |
| Imbalanced-learn | SMOTE for class imbalance |
| XGBoost | Gradient boosting model |
| TensorFlow/Keras | Neural network experiments |
| Streamlit | Web app deployment |
| Git/GitHub | Version control |

---

## 📁 Repository Structure

banking-crisis-app/
├── app.py                 # Streamlit web application
├── best_model.pkl         # Trained Random Forest model
├── scaler.pkl             # Fitted StandardScaler
├── requirements.txt       # Python dependencies
├── MODEL_CARD.md          # Model documentation
└── README.md              # This file

---

## ⚠️ Limitations

- Data is historical (up to 2014) — may not reflect current conditions
- Trained on 13 countries — may not generalise to all African nations
- Class imbalance mitigated with SMOTE but crisis cases still rare
- Should supplement, not replace, professional economic analysis

---

## 🔮 Next Steps

- Retrain with post-2014 data as it becomes available
- Add more African countries to improve generalisability
- Integrate real-time economic data feeds
- Build explainability layer (SHAP values) for policymaker trust

---

## 👩‍💻 Author

**Tracy Aumo**  
Africa AI Hub — AI Skills Immersion Programme (AISIP) Cohort 1  
Pathway 4: AI Engineering  

---

## 📋 Model Card

See [MODEL_CARD.md](MODEL_CARD.md) for full model documentation
including intended use, limitations, and ethical considerations.


