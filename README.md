# 🛒 E-Commerce Capstone — 5 ML Models

Machine learning capstone on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (~100K real orders). One shared data pipeline, five models covering classification, regression, and clustering.

## Models

| # | Model | Task | Target |
|---|-------|------|--------|
| 1 | Return Prediction | Classification | `is_canceled` |
| 2 | Delivery Delay Prediction | Classification | `is_late` |
| 3 | Customer Rating Prediction | Regression | `review_score` |
| 4 | Customer Segmentation | Clustering | RFM features |
| 5 | Revenue Prediction | Regression | `net_revenue` |

Each model: preprocessing pipeline → compare 4 algorithms → tune the best one → evaluate → export.

## How to Use

**1. Install dependencies**
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib kagglehub jupyter
```

**2. Open and run the notebook**
```bash
jupyter notebook Capstone_Models_1_to_5.ipynb
```
Dataset downloads automatically on the first cell (needs a free Kaggle account for `kagglehub`). Run top to bottom — `RANDOM_STATE = 42` makes results reproducible.

**3. Use a trained model without retraining**
```python
import joblib

model = joblib.load("model/model1_best_return_prediction.pkl")
predictions = model.predict(X_new)
```

## Repo Structure

```
├── Capstone_Models_1_to_5.ipynb      # notebook — all 5 models
├── Capstone_ML_Project_Report.docx   # written report
├── data/                             # cleaned datasets
├── model/                            # trained models (.pkl)
└── reports/                          # metrics & comparison tables (.csv)
```

## Notes

- Olist has no `Returned`, `age_group`, or `discount` fields — closest available columns are used as documented proxies (e.g. `is_canceled` for `Returned`).
- Imbalanced targets → F1-macro and ROC-AUC used over Accuracy.
- Feature importance shows correlation, not causation.

---
**Dataset:** Olist Brazilian E-Commerce · **Random State:** 42
