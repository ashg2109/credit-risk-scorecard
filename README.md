# Credit Risk Scorecard
### Machine Learning model to predict loan default probability

## Overview
Built a credit risk classification model on 1,000 German credit applicants, 
comparing Logistic Regression, Random Forest, and XGBoost to predict default risk.

## Results
| Model | AUC-ROC |
|---|---|
| Logistic Regression | 0.786 |
| Random Forest | 0.785 |
| XGBoost | 0.759 |

## Key Findings
- **Checking account status** is the strongest predictor of default
- **Loan duration > 24 months** significantly increases risk
- **Low savings + no checking balance** = 2.3x higher default probability
- Logistic Regression outperformed XGBoost — simpler models can win on well-structured smaller datasets

## Tech Stack
Python · Pandas · Scikit-learn · XGBoost · SHAP · Matplotlib

## Files
- `credit_risk.ipynb` — full analysis notebook
- `eda_charts.png` — exploratory data analysis
- `shap_summary.png` — SHAP feature importance
- `roc_curve.png` — model comparison

## Dataset
German Credit Data — 1,000 applicants, 9 features, 30% default rate
