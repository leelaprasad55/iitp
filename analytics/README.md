# Analytics and Machine Learning

This module uses the Titanic dataset.

## Important data rule

`01_eda.ipynb` loads Titanic **once** with `sns.load_dataset("titanic")` and immediately saves `analytics/titanic.csv`. `02_modeling.ipynb` and later EDA cells load that CSV only.

## Files

| Path | Role |
| --- | --- |
| `01_eda.ipynb` | Profiling, missingness rules, univariate, bivariate, correlation, multivariate, z-score demo |
| `02_modeling.ipynb` | Classification, imbalance, GridSearch, OOB, fare regression, joblib pipeline |
| `titanic.csv` | Saved snapshot of the seaborn Titanic dataset |
| `outputs/charts/` | Saved figures |
| `outputs/metrics/` | Saved metric tables |
| `models/best_pipeline.joblib` | Fitted preprocess + estimator |

## Run

From the repository root, after installing `requirements.txt`:

```bash
jupyter notebook analytics/01_eda.ipynb
jupyter notebook analytics/02_modeling.ipynb
```

Or execute non-interactively:

```bash
jupyter nbconvert --to notebook --execute analytics/01_eda.ipynb --inplace
jupyter nbconvert --to notebook --execute analytics/02_modeling.ipynb --inplace
```
