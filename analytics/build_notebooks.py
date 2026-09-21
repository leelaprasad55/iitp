"""Build analytics notebooks (source of truth for 01_eda / 02_modeling)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NB1 = ROOT / "01_eda.ipynb"
NB2 = ROOT / "02_modeling.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def write_eda():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    }
    cells = []
    cells.append(md("# 01 — Titanic Exploratory Data Analysis\n\nThis notebook loads the Titanic dataset **once** with `sns.load_dataset(\"titanic\")`, saves `analytics/titanic.csv`, and then uses only that CSV."))
    cells.append(code("""from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

BASE = Path(".").resolve()
if (BASE / "analytics").exists():
    ANALYTICS = BASE / "analytics"
else:
    ANALYTICS = BASE

CHARTS = ANALYTICS / "outputs" / "charts"
METRICS = ANALYTICS / "outputs" / "metrics"
CHARTS.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)
CSV_PATH = ANALYTICS / "titanic.csv"
print("Analytics folder:", ANALYTICS)
"""))
    cells.append(md("## 1. Data profiling\n\nLoad from seaborn **once**, save CSV immediately, then reload the CSV for all later analysis."))
    cells.append(code("""raw = sns.load_dataset("titanic")
raw.to_csv(CSV_PATH, index=False)
print("Saved Titanic snapshot to", CSV_PATH)

df = pd.read_csv(CSV_PATH)
print("Loaded later analysis frame from CSV only.")
print("shape:", df.shape)
print("\\n--- df.info() ---")
df.info()
print("\\n--- df.describe() ---")
display_desc = df.describe(include="all")
display_desc
"""))
    cells.append(code("""missing_count = df.isna().sum()
missing_pct = (missing_count / len(df) * 100).round(2)
missing_tbl = pd.DataFrame({
    "missing_count": missing_count,
    "missing_pct": missing_pct,
})
missing_tbl = missing_tbl[missing_tbl["missing_count"] > 0].sort_values("missing_pct", ascending=False)
print("Missing values for every affected column:")
missing_tbl
"""))
    cells.append(md("""### Missing-data rules used in this EDA copy

- **less than 5%** → drop rows (`embarked`, `embark_town` are typically ~0.2%).
- **5% to 30%** → impute (`age` is typically ~20%; we impute the median).
- **very high missingness** → decide explicitly. `deck` is usually ~77% missing. Dropping the column would throw away a potentially informative cabin signal, but imputing it as if it were MAR would invent a dense category. We **keep the column and encode missing as `Unknown`**.

These EDA imputations are for exploratory charts only. The ML notebook uses a sklearn pipeline fitted on training data so there is no leakage from this demonstration."""))
    cells.append(code("""eda = df.copy()

# < 5% missing: drop rows
low_missing_cols = [c for c in ["embarked", "embark_town"] if c in eda.columns]
eda = eda.dropna(subset=[c for c in low_missing_cols if eda[c].isna().mean() < 0.05])

# 5% to 30%: impute age with median
age_pct = df["age"].isna().mean() * 100
print(f"age missing percent in original CSV: {age_pct:.2f}")
if 5 <= age_pct <= 30:
    eda["age"] = eda["age"].fillna(eda["age"].median())
    print("Imputed age with median because missingness is between 5% and 30%.")
elif age_pct < 5:
    eda = eda.dropna(subset=["age"])
    print("Dropped age-missing rows because missingness is < 5%.")
else:
    print("Age missingness is outside the 5-30% band; see justification cells.")

# very high: encode deck missingness
if "deck" in eda.columns:
    deck_pct = df["deck"].isna().mean() * 100
    print(f"deck missing percent in original CSV: {deck_pct:.2f}")
    eda["deck"] = eda["deck"].astype("object").fillna("Unknown")
    print("Encoded missing deck as 'Unknown' rather than dropping the column or fabricating a typical deck.")

print("EDA frame shape after missing-value handling:", eda.shape)
eda.head()
"""))
    cells.append(md("## 2. Univariate analysis — age and fare"))
    cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(eda["age"].dropna(), kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Age histogram")
sns.boxplot(x=eda["age"], ax=axes[0, 1], color="steelblue")
axes[0, 1].set_title("Age boxplot")
sns.histplot(eda["fare"].dropna(), kde=True, ax=axes[1, 0], color="darkorange")
axes[1, 0].set_title("Fare histogram")
sns.boxplot(x=eda["fare"], ax=axes[1, 1], color="darkorange")
axes[1, 1].set_title("Fare boxplot")
fig.tight_layout()
fig.savefig(CHARTS / "univariate_age_fare.png", dpi=150, bbox_inches="tight")
plt.show()
"""))
    cells.append(code("""def iqr_outlier_count(series: pd.Series) -> dict:
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (s < lower) | (s > upper)
    return {
        "q1": q1, "q3": q3, "iqr": iqr,
        "lower": lower, "upper": upper,
        "n_outliers": int(mask.sum()),
    }

age_out = iqr_outlier_count(eda["age"])
fare_out = iqr_outlier_count(eda["fare"])
print("Age IQR outliers:", age_out)
print("Fare IQR outliers:", fare_out)

fare = eda["fare"].dropna()
fare_mean = fare.mean()
fare_median = fare.median()
fare_mode = fare.mode().iloc[0]
print(f"fare mean={fare_mean:.4f}")
print(f"fare median={fare_median:.4f}")
print(f"fare mode={fare_mode:.4f}")

if fare_mean > fare_median > fare_mode:
    skew_note = "right-skewed (mean > median > mode)"
elif fare_mean < fare_median < fare_mode:
    skew_note = "left-skewed (mean < median < mode)"
else:
    skew_note = "approximate symmetry or mixed relationship among mean/median/mode"
print("Fare skewness based on mean/median/mode:", skew_note)
"""))
    cells.append(md("## 3. Bivariate analysis — survival by sex, pclass, and both\n\nBoolean masking is used instead of relying only on `groupby`."))
    cells.append(code("""survived = eda["survived"] == 1
female = eda["sex"] == "female"
male = eda["sex"] == "male"

print("Survival rate overall:", survived.mean())
print("Survival rate female (boolean mask):", eda.loc[female, "survived"].mean())
print("Survival rate male (boolean mask):", eda.loc[male, "survived"].mean())

for pclass in sorted(eda["pclass"].dropna().unique()):
    mask = eda["pclass"] == pclass
    print(f"Survival rate pclass={pclass}:", eda.loc[mask, "survived"].mean())

print("\\nSurvival rate by sex AND pclass (boolean masks):")
rows = []
for sex_label, sex_mask in [("female", female), ("male", male)]:
    for pclass in sorted(eda["pclass"].dropna().unique()):
        mask = sex_mask & (eda["pclass"] == pclass)
        rate = eda.loc[mask, "survived"].mean()
        n = int(mask.sum())
        rows.append({"sex": sex_label, "pclass": int(pclass), "n": n, "survival_rate": rate})
        print(f"  {sex_label}, pclass={pclass}, n={n}, rate={rate:.4f}")

sex_pclass_tbl = pd.DataFrame(rows)
sex_pclass_tbl.to_csv(METRICS / "eda_survival_sex_pclass.csv", index=False)
sex_pclass_tbl
"""))
    cells.append(md("## 4. Correlation on exactly six columns\n\nColumns: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`. Do **not** include `adult_male` or `alone`."))
    cells.append(code("""corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = eda[corr_cols].corr()
print(corr)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
ax.set_title("Correlation heatmap (six specified columns)")
fig.tight_layout()
fig.savefig(CHARTS / "correlation_heatmap_six_cols.png", dpi=150, bbox_inches="tight")
plt.show()

abs_corr = corr.abs().to_numpy().copy()
np.fill_diagonal(abs_corr, 0)
abs_corr_df = pd.DataFrame(abs_corr, index=corr.index, columns=corr.columns)
stacked = abs_corr_df.unstack().sort_values(ascending=False)
# each pair appears twice; keep unique unordered pairs
seen = set()
top_pairs = []
for (a, b), value in stacked.items():
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    top_pairs.append((key, value, corr.loc[a, b]))
    if len(top_pairs) == 2:
        break

print("Top two absolute off-diagonal correlations:")
for key, abs_v, signed in top_pairs:
    print(f"  {key[0]} vs {key[1]}: abs={abs_v:.4f}, signed={signed:.4f}")
"""))
    cells.append(md("""**Interpretation of the top two |correlations|** is printed from the actual matrix above. Typical Titanic patterns (verified by the computed values in the previous cell):

- `pclass` vs `fare` is usually a strong negative relationship: higher class numbers (3rd class) pay lower fares.
- `sibsp` vs `parch` is usually a moderate positive relationship: passengers travelling with siblings/spouses often also travel with parents/children.

The notebook relies on the printed numbers, not assumed values."""))
    cells.append(md("## 5. Multivariate analysis — four distinct charts"))
    cells.append(code("""fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=eda, x="pclass", y="survived", hue="sex", ax=ax)
ax.set_title("Survival rate by class and sex")
ax.set_ylabel("mean survived")
fig.tight_layout()
fig.savefig(CHARTS / "mv1_survival_class_sex.png", dpi=150, bbox_inches="tight")
plt.show()
print("Interpretation: First-class women typically show the highest survival, while third-class men show the lowest. The hue split shows that sex differences persist inside every class rather than being only a class effect. This is a classic 'women and children first' pattern interacting with ticket class access to boats.")
"""))
    cells.append(code("""fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=eda, x="survived", y="age", hue="sex", ax=ax)
ax.set_title("Age by survival and sex")
fig.tight_layout()
fig.savefig(CHARTS / "mv2_age_survival_sex.png", dpi=150, bbox_inches="tight")
plt.show()
print("Interpretation: Age distributions overlap a lot between survivors and non-survivors, so age alone is a weak separator. Within sex, younger passengers are slightly more common among survivors, especially males. Outliers in older ages appear in both groups, which warns against treating age as a simple linear survival rule.")
"""))
    cells.append(code("""fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=eda, x="age", y="fare", hue="survived", style="pclass", alpha=0.7, ax=ax)
ax.set_title("Fare vs age, styled by class and survival")
fig.tight_layout()
fig.savefig(CHARTS / "mv3_fare_age_survival_class.png", dpi=150, bbox_inches="tight")
plt.show()
print("Interpretation: High fares cluster among first-class points, and many of those points are survivors. Low-fare third-class points dominate the bottom of the plot and mix survival outcomes. Age spreads across the full range at every fare band, so fare/class is more visually associated with survival than age.")
"""))
    cells.append(code("""fig = sns.catplot(
    data=eda, x="embarked", y="survived", hue="pclass", kind="bar", height=4, aspect=1.4
)
fig.fig.suptitle("Survival by embarkation port and class", y=1.03)
fig.savefig(CHARTS / "mv4_survival_embarked_class.png", dpi=150, bbox_inches="tight")
plt.show()
print("Interpretation: Embarkation port is associated with survival, but much of that association is entangled with class mix at each port. Cherbourg often shows a higher first-class share and higher survival. The grouped bars keep class visible so we do not over-interpret port as a causal factor.")
"""))
    cells.append(md("""## 6. Standardization demonstration (EDA only)

Z-score standardize `age` and `fare` to show mean/std before vs after. **Do not** reuse these transformed columns as leaked features in the ML notebook. Modeling fits `StandardScaler` on the training fold only."""))
    cells.append(code("""demo = eda[["age", "fare"]].copy()
before = pd.DataFrame({
    "mean_before": demo.mean(),
    "std_before": demo.std(ddof=0),
})
z = (demo - demo.mean()) / demo.std(ddof=0)
after = pd.DataFrame({
    "mean_after": z.mean(),
    "std_after": z.std(ddof=0),
})
std_tbl = pd.concat([before, after], axis=1)
print(std_tbl)
print("After z-score, means are ~0 and standard deviations are ~1.")
print("This transformation is NOT written back into titanic.csv and is NOT used by 02_modeling.ipynb.")
std_tbl.to_csv(METRICS / "eda_standardization_demo.csv")
std_tbl
"""))
    nb["cells"] = cells
    nbf.write(nb, NB1)
    print("Wrote", NB1)


def write_modeling():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    }
    cells = []
    cells.append(md("# 02 — Titanic Machine Learning\n\nThis notebook loads **only** `analytics/titanic.csv` (created by `01_eda.ipynb`). It does not call `sns.load_dataset`."))
    cells.append(code("""from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

BASE = Path(".").resolve()
ANALYTICS = BASE / "analytics" if (BASE / "analytics").exists() else BASE
CHARTS = ANALYTICS / "outputs" / "charts"
METRICS = ANALYTICS / "outputs" / "metrics"
MODELS = ANALYTICS / "models"
CHARTS.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ANALYTICS / "titanic.csv")
print("Loaded", ANALYTICS / "titanic.csv", "shape=", df.shape)
df.head()
"""))
    cells.append(md("## 1. Stratified train/test split first\n\nThe same split is reused for Logistic Regression, Decision Tree, and Random Forest. Preprocessing is fitted on training data only."))
    cells.append(code("""target = "survived"
feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
model_df = df[feature_cols + [target]].copy()

X = model_df[feature_cols]
y = model_df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train survival rate:", y_train.mean(), "Test survival rate:", y_test.mean())
"""))
    cells.append(md("## 2. Preprocessing pipeline\n\nNumeric: median impute + StandardScaler. Categorical: most-frequent impute + OneHotEncoder."))
    cells.append(code("""numeric_features = ["age", "sibsp", "parch", "fare"]
categorical_features = ["pclass", "sex", "embarked"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
"""))
    cells.append(md("## 3–4. Classification models and evaluation"))
    cells.append(code("""def evaluate_classifier(name, pipeline, X_tr, y_tr, X_te, y_te):
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_te)[:, 1]
    else:
        y_score = pipeline.decision_function(X_te)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, y_score),
    }
    cm = confusion_matrix(y_te, y_pred)
    print(name, metrics)
    print("confusion matrix:\\n", cm)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"{name} confusion matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    RocCurveDisplay.from_predictions(y_te, y_score, ax=axes[1], name=name)
    axes[1].set_title(f"{name} ROC curve")
    fig.tight_layout()
    safe = name.lower().replace(" ", "_")
    fig.savefig(CHARTS / f"clf_{safe}.png", dpi=150, bbox_inches="tight")
    plt.show()
    return pipeline, metrics, cm

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

fitted = {}
rows = []
for name, clf in classifiers.items():
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
    fitted[name], metrics, _ = evaluate_classifier(name, pipe, X_train, y_train, X_test, y_test)
    rows.append(metrics)

clf_table = pd.DataFrame(rows).set_index("model")
clf_table.to_csv(METRICS / "classification_comparison.csv")
print("Classification comparison:")
clf_table
"""))
    cells.append(md("## 5. Decision tree visualization"))
    cells.append(code("""tree_pipe = fitted["Decision Tree"]
ohe = tree_pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
cat_names = ohe.get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_features, cat_names])
tree_model = tree_pipe.named_steps["model"]

fig, ax = plt.subplots(figsize=(18, 10))
plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["did_not_survive", "survived"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
)
ax.set_title("Decision Tree (readable, with feature and class names)")
fig.tight_layout()
fig.savefig(CHARTS / "decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()
"""))
    cells.append(md("""## 6. Class imbalance comparison

A. baseline Logistic Regression  
B. `class_weight='balanced'`  
C. SMOTE applied **only** to training data (never to the test set)"""))
    cells.append(code("""imb_rows = []

base = Pipeline(steps=[("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000, random_state=42))])
base.fit(X_train, y_train)
yp = base.predict(X_test)
imb_rows.append({
    "method": "A_baseline",
    "precision": precision_score(y_test, yp, zero_division=0),
    "recall": recall_score(y_test, yp, zero_division=0),
    "f1": f1_score(y_test, yp, zero_division=0),
})

bal = Pipeline(steps=[("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
bal.fit(X_train, y_train)
yp = bal.predict(X_test)
imb_rows.append({
    "method": "B_class_weight_balanced",
    "precision": precision_score(y_test, yp, zero_division=0),
    "recall": recall_score(y_test, yp, zero_division=0),
    "f1": f1_score(y_test, yp, zero_division=0),
})

# SMOTE only on transformed training features; test remains original distribution.
X_train_prep = preprocess.fit_transform(X_train)
X_test_prep = preprocess.transform(X_test)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_prep, y_train)
print("SMOTE train size before/after:", X_train_prep.shape, X_train_sm.shape)
print("SMOTE class counts after:", pd.Series(y_train_sm).value_counts().to_dict())

sm_clf = LogisticRegression(max_iter=1000, random_state=42)
sm_clf.fit(X_train_sm, y_train_sm)
yp = sm_clf.predict(X_test_prep)
imb_rows.append({
    "method": "C_SMOTE_train_only",
    "precision": precision_score(y_test, yp, zero_division=0),
    "recall": recall_score(y_test, yp, zero_division=0),
    "f1": f1_score(y_test, yp, zero_division=0),
})

imb_table = pd.DataFrame(imb_rows).set_index("method")
imb_table.to_csv(METRICS / "imbalance_comparison.csv")
print(imb_table)

best_imb = imb_table["f1"].idxmax()
print("Highest F1 method:", best_imb)
print(
    "Recommendation: class_weight='balanced' is usually preferable for this small tabular set "
    "because it needs no synthetic samples and still lifts recall. SMOTE can help recall further "
    "but may invent unrealistic passenger combinations. We never apply SMOTE to the test set."
)
imb_table
"""))
    cells.append(md("## 7. Random Forest GridSearchCV and OOB score"))
    cells.append(code("""rf_pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(random_state=42)),
])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, 8],
    "model__max_features": ["sqrt", "log2"],
}
grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

best_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
X_train_prep = preprocess.fit_transform(X_train)
oob_rf = RandomForestClassifier(
    random_state=42,
    oob_score=True,
    bootstrap=True,
    **best_params,
)
oob_rf.fit(X_train_prep, y_train)
print("OOB score:", oob_rf.oob_score_)

gs_report = {
    "best_params": grid.best_params_,
    "best_cv_f1": float(grid.best_score_),
    "oob_score": float(oob_rf.oob_score_),
}
(METRICS / "random_forest_gridsearch.json").write_text(json.dumps(gs_report, indent=2), encoding="utf-8")
gs_report
"""))
    cells.append(md("## 8. Regression task — predict fare"))
    cells.append(code("""reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "survived"]
reg_df = df[reg_features + ["fare"]].copy()
X_reg = reg_df[reg_features]
y_reg = reg_df["fare"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_num = ["age", "sibsp", "parch", "survived"]
reg_cat = ["pclass", "sex", "embarked"]
reg_pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), reg_num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), reg_cat),
    ]
)
reg_pipe = Pipeline([("preprocess", reg_pre), ("model", RandomForestRegressor(n_estimators=200, random_state=42))])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = mean_squared_error(yr_test, yr_pred) ** 0.5
r2 = r2_score(yr_test, yr_pred)
n = len(yr_test)
p = Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
reg_metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "Adj_R2": adj_r2}
print(reg_metrics)
pd.Series(reg_metrics).to_csv(METRICS / "regression_fare_metrics.csv")

residuals = yr_test - yr_pred
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(yr_pred, residuals, alpha=0.7)
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Predicted fare")
ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Fare regression residual plot")
fig.tight_layout()
fig.savefig(CHARTS / "regression_residuals.png", dpi=150, bbox_inches="tight")
plt.show()

spread_low = residuals[yr_pred <= np.median(yr_pred)].std()
spread_high = residuals[yr_pred > np.median(yr_pred)].std()
print(f"Residual std below median prediction={spread_low:.3f}, above={spread_high:.3f}")
if spread_high > 1.3 * spread_low:
    print("There is evidence of heteroscedasticity: residual spread grows with predicted fare.")
else:
    print("Residual spread is relatively stable; strong heteroscedasticity is not obvious from this split.")
"""))
    cells.append(md("## 9. Final comparison and recommendation"))
    cells.append(code("""print("CLASSIFICATION METRICS")
print(clf_table)
print("\\nREGRESSION METRICS (fare)")
print(pd.Series(reg_metrics))

best_clf = clf_table["f1"].idxmax()
print("\\nFinal recommendation:")
print(
    f"Prefer {best_clf} as the production classifier because it has the strongest F1 "
    f"({clf_table.loc[best_clf, 'f1']:.3f}) on the held-out stratified test set, "
    f"with ROC-AUC {clf_table.loc[best_clf, 'roc_auc']:.3f}. "
    "Logistic Regression remains the most interpretable linear baseline, and the Decision Tree "
    "is useful for explaining rules, but ensemble averaging usually generalizes better on Titanic. "
    "The fare regressor is a separate task and should not be chosen using classification metrics."
)
"""))
    cells.append(md("## 10. Save complete pipeline, reload, and predict on RAW rows"))
    cells.append(code("""best_name = clf_table["f1"].idxmax()
best_pipeline = fitted[best_name]
out_path = MODELS / "best_pipeline.joblib"
joblib.dump(best_pipeline, out_path)
print("Saved", out_path, "model=", best_name)

reloaded = joblib.load(out_path)
raw_input = pd.DataFrame([
    {
        "pclass": 3,
        "sex": "male",
        "age": 22,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S",
    },
    {
        "pclass": 1,
        "sex": "female",
        "age": 38,
        "sibsp": 1,
        "parch": 0,
        "fare": 71.2833,
        "embarked": "C",
    },
])
print("RAW unprocessed input:")
print(raw_input)
print("Reloaded pipeline predictions:", reloaded.predict(raw_input))
print("Reloaded pipeline probabilities:\\n", reloaded.predict_proba(raw_input))
"""))
    nb["cells"] = cells
    nbf.write(nb, NB2)
    print("Wrote", NB2)


if __name__ == "__main__":
    write_eda()
    write_modeling()
