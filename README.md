# Telecom Customer Churn Prediction

## Business Problem
A telecom company loses thousands of subscribers monthly. Acquiring new customers costs 5-7x more than retaining existing ones. This project builds a machine learning system to identify high-risk customers before they churn — enabling proactive retention strategies.

## Dataset
- **Source**: IBM Telco Customer Churn (Kaggle)
- **Size**: 7,043 customers × 21 features
- **Target**: Churn (Yes/No → 1/0)

## Models Trained
| Model | ROC-AUC | Recall | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.8290 | 0.6711 | 0.5892 |
| Random Forest | 0.8208 | 0.6310 | 0.5871 |
| XGBoost | 0.8141 | 0.6444 | 0.5828 |

## Best Model
**Logistic Regression** — ROC-AUC: 0.8290

## SHAP Explainability
Top churn drivers:
1. TotalServices — multi-service customers at risk
2. MonthlyCharges — high bills drive churn
3. InternetService Fiber — fiber users churn more
4. tenure — new customers most vulnerable
5. PaymentMethod — electronic check users at risk

## Business Recommendations
1. Convert month-to-month customers to annual contracts
2. Create onboarding program for first 12 months
3. Introduce bundle pricing for multi-service customers
4. Proactively target 251 high-risk customers monthly
5. Review fiber optic pricing strategy

## Business Impact
- Model catches 251 churners per month
- Retention cost $50 vs Acquisition cost $300
- **Monthly savings: $62,750**
- **Annual savings: $753,000**

##  Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, SMOTE

##  Author
**Sineka** — Data Scientist
[GitHub](https://github.com/sineka-ds)



