📉 Customer Churn Prediction — Logistic Regression

A complete, interpretable machine learning pipeline that predicts telecom customer churn using Logistic Regression — built with a focus on statistical rigor (multicollinearity checks, odds-ratio interpretation) and business-actionable insights, not just raw accuracy.

🎯 Problem Statement

Customer churn (customers leaving a service) is one of the costliest problems in subscription-based businesses — acquiring a new customer typically costs far more than retaining an existing one. This project builds a classifier that predicts which customers are likely to churn, and — more importantly — explains why, so retention efforts can be targeted effectively.

📊 Dataset
Source: IBM Telco Customer Churn Dataset
Size: 7,043 customers × 21 features
Target: Churn (Yes/No) — 26.5% positive class (moderately imbalanced)
Features: Demographics (gender, senior citizen, partner, dependents), account info (tenure, contract type, payment method), services subscribed (internet, phone, streaming, security add-ons), and billing (monthly/total charges)
🔧 Pipeline
Stage	What was done
EDA	Target balance check, correlation exploration, discovered a hidden data quality issue in TotalCharges
Data Cleaning	Fixed TotalCharges (blank strings → numeric, tied to tenure=0 new customers), dropped identifier column
Feature Engineering	Collapsed redundant categorical levels ("No internet service" → "No"), binary Yes/No → 0/1, one-hot encoding with drop_first=True to avoid the dummy variable trap
Train/Test Split	80/20 stratified split to preserve class ratio
Scaling	StandardScaler fit on training data only (prevents data leakage)
Baseline Model	Logistic Regression with class_weight='balanced' to handle class imbalance
Multicollinearity Check	Variance Inflation Factor (VIF) analysis — flagged and removed TotalCharges (VIF ≈ 11, redundant with tenure × MonthlyCharges)
Model Interpretation	Converted coefficients to odds ratios for business-readable insights
Threshold Tuning	Precision-recall tradeoff analysis; explored F1-optimal and Youden's J thresholds vs. business-cost-based selection
Regularization	Compared L1 (Lasso, automatic feature selection) vs. L2 (Ridge, default)
📈 Results
Metric	Score
ROC-AUC	0.84
Recall (Churn class)	0.79
Precision (Churn class)	0.51
F1 Score	0.62

Recall was prioritized over precision via class_weight='balanced', since missing an actual churner (false negative) is costlier to the business than a false retention-offer alarm (false positive).

💡 Key Business Insights (from Odds Ratios)
Factor	Effect on Churn Odds
Fiber optic internet	+206% vs. DSL
Two-year contract	−75%
One-year contract	−51%
Higher tenure	−53% per std. deviation increase
Electronic check payment	+49%
Online Security add-on	−29%

These aren't just predictions — they're actionable levers: e.g., converting a month-to-month customer to a one-year contract is associated with a ~51% drop in churn odds.

🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · statsmodels · matplotlib
