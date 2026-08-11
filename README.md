# Loan Default Risk Prediction

A machine learning system that predicts credit default risk on an imbalanced
dataset, with a focus on **business-cost-aware decision making** and
**explainability** — not just raw accuracy.

🔗 **Live demo:** https://lalitchoudhary-cloud-spcmruu29spruetyfmnfxc.streamlit.app/
📓 **Notebook:** https://www.kaggle.com/code/lalit786/notebook95887812de

---

## Problem

Given a borrower's credit history, demographics, and 6 months of billing/
repayment data, predict whether they will default on their next payment.
The dataset is moderately imbalanced (~22% default rate), so the project
treats this as a **cost-sensitive decision problem**, not just a
classification exercise — a missed default is far more expensive to a
lender than a wrongly-rejected good customer, and the model/threshold are
chosen with that asymmetry in mind.

## Dataset

[UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
(24,000 rows, 24 features: credit limit, demographics, 6 months of
repayment status, bill amounts, and payment amounts).

## Approach

**1. EDA & Cleaning**
- Removed 24 duplicate rows, fixed undocumented category codes in
  `EDUCATION`/`MARRIAGE` (values outside the official 1–4 / 1–3 codebook)
- Identified `PAY_1` (recent repayment delay) and `LIMIT_BAL` (credit
  limit) as the strongest raw predictors

**2. Feature Engineering**
- 8 domain-justified derived features: credit utilization ratio, average
  payment ratio, average/max/worsening repayment delay trend, age bucket
- Each validated for correlation with the target before being kept —
  `max_delay` alone (0.33 correlation) outperforms any single raw `PAY_*`
  column

**3. Modeling**
- Compared Logistic Regression, Random Forest, and XGBoost, each with and
  without imbalance handling (`class_weight`, SMOTE, `scale_pos_weight`)
- **Finding:** imbalance-handling techniques barely moved ROC-AUC/PR-AUC —
  they mainly shifted the precision/recall tradeoff at a fixed 0.5
  threshold. Plain Random Forest had the best ranking ability of everything
  tested.
- Hyperparameter-tuned the Random Forest via `RandomizedSearchCV` (3-fold
  stratified CV, optimizing PR-AUC)

**4. Cost-Matrix Threshold Selection** *(the core contribution)*
- Instead of using the default 0.5 threshold or resampling, computed the
  **cost-optimal decision threshold** using a business cost matrix (missing
  a default ≈ 6.7x more expensive than wrongly rejecting a good customer)
- Result: shifting the threshold from 0.5 to **0.15** reduced estimated
  total cost by **35.6%** on the validation set

**5. Explainability & Fairness**
- SHAP global feature importance + individual waterfall explanations for a
  true positive and a false positive
- Fairness check on `SEX`: recall and false-positive rate are close across
  groups (0.778 vs 0.804 recall; 0.422 vs 0.402 FPR), and `SEX` ranks 30th
  of 34 features in importance — the default-rate gap seen in EDA reflects
  real underlying risk factors, not direct reliance on the attribute

**6. Deployment**
- Streamlit app: enter borrower details → get a risk score, risk tier,
  approve/reject decision, and a live SHAP explanation

## Results (held-out test set, never touched until final evaluation)

| Metric | Value |
|---|---|
| ROC-AUC | 0.793 |
| PR-AUC | 0.577 |
| Precision @ threshold 0.15 | 0.35 |
| Recall @ threshold 0.15 | 0.79 |
| F1 @ threshold 0.15 | 0.49 |

At the cost-optimal threshold, the model catches **79% of actual
defaulters**, deliberately trading precision for recall since the cost
matrix says that's the right call.

## Key takeaway

The most defensible modeling decision in this project wasn't a fancier
algorithm — it was recognizing that **imbalance-handling techniques don't
improve a model's ranking ability, they just move the operating point**,
and that the operating point should be chosen from business cost, not
convention. A well-tuned threshold beat every resampling strategy tried.

## Repo structure

```
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
├── data/
│   ├── credit.csv                 # raw data
│   └── credit_features.csv        # after feature engineering
├── app/
│   ├── app.py                     # Streamlit app
│   ├── model.pkl                  # trained model
│   ├── model_metadata.json        # feature columns, thresholds
│   └── requirements.txt
└── README.md
```

## Running the app locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn · SHAP ·
Streamlit

---

*Dataset source: UCI Machine Learning Repository, "Default of Credit Card
Clients Dataset" (Yeh, I. C., 2016). Used for educational/portfolio
purposes.*
