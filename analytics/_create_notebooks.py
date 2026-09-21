"""One-shot notebook builder. Run from repo root: python analytics/_create_notebooks.py"""
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
    cells.append(md("# Titanic EDA\n\nExploratory analysis for the Zepto Data & AI Platform.\n\n**Rule:** load Seaborn's Titanic dataset **once**, save `analytics/titanic.csv`, then use only that CSV."))
    cells.append(code("""from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE = Path(".").resolve()
if not (BASE / "01_eda.ipynb").exists() and (BASE / "analytics" / "01_eda.ipynb").exists():
    BASE = BASE / "analytics"
elif (BASE / "analytics").exists() and not (BASE / "01_eda.ipynb").exists():
    BASE = BASE / "analytics"

CHARTS = BASE / "outputs" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)
CSV_PATH = BASE / "titanic.csv"
print("Working directory:", BASE)
"""))
    cells.append(md("## 1. Data profiling\n\nLoad Titanic **once** with `sns.load_dataset('titanic')` and immediately persist it."))
    cells.append(code("""raw = sns.load_dataset("titanic")
raw.to_csv(CSV_PATH, index=False)
print("Saved Titanic once to", CSV_PATH)

# All later analysis uses the saved CSV, not another independent load.
df = pd.read_csv(CSV_PATH)
print("Reloaded from CSV. Shape:", df.shape)
df.head()
"""))
    cells.append(code("""print("=== df.info() ===")
df.info()
print("\\n=== df.describe() ===")
display_desc = df.describe(include="all")
display_desc
"""))
    cells.append(code("""print("shape:", df.shape)
missing_count = df.isna().sum()
missing_pct = (missing_count / len(df) * 100).round(2)
missing_table = pd.DataFrame({
    "missing_count": missing_count,
    "missing_pct": missing_pct,
})
affected = missing_table[missing_table["missing_count"] > 0].sort_values("missing_pct", ascending=False)
print("Missing values for every affected column:")
affected
"""))
    cells.append(md("""### Missing-data rules used in this notebook

- **< 5% missing:** drop those rows (small loss of sample size).
- **5% to 30% missing:** impute (keep the rows, fill a reasonable central value / mode).
- **Very high missingness:** do **not** blindly impute. Either drop the column or encode missing as its own category, with an explicit reason.

Typical Titanic pattern (computed above, not assumed):

- `embarked` / `embark_town` are usually well under 5% → **drop those few rows**.
- `age` is usually in the 5–30% band → **impute with the median** (robust to fare/age outliers).
- `deck` is usually >70% missing → **encode missing as `'Unknown'`** instead of dropping the column or inventing a deck. Deck is still a useful signal when it *is* recorded (often first-class passengers), so throwing the column away would discard information. Median/mode imputation would be misleading because most values are simply not observed.
"""))
    cells.append(code("""eda = df.copy()

missing_pct_series = eda.isna().mean() * 100
print("Missing % by column:\\n", missing_pct_series.round(2))

# < 5%: drop rows for those columns
low_missing_cols = [c for c in eda.columns if 0 < missing_pct_series[c] < 5]
print("Low-missing columns (drop rows):", low_missing_cols)
if low_missing_cols:
    before = len(eda)
    eda = eda.dropna(subset=low_missing_cols)
    print(f"Dropped {before - len(eda)} rows due to <5% missingness in {low_missing_cols}.")

# 5% to 30%: impute
mid_missing_cols = [c for c in eda.columns if 5 <= missing_pct_series[c] <= 30]
print("Mid-missing columns (impute):", mid_missing_cols)
for col in mid_missing_cols:
    if pd.api.types.is_numeric_dtype(eda[col]):
        value = eda[col].median()
        eda[col] = eda[col].fillna(value)
        print(f"Imputed numeric {col} with median={value:.4f}")
    else:
        value = eda[col].mode().iloc[0]
        eda[col] = eda[col].fillna(value)
        print(f"Imputed categorical {col} with mode={value}")

# Very high missingness: encode missing
high_missing_cols = [c for c in eda.columns if missing_pct_series[c] > 30]
print("High-missing columns (encode or drop with justification):", high_missing_cols)
for col in high_missing_cols:
    if col == "deck":
        eda[col] = eda[col].astype("object").fillna("Unknown")
        print("Encoded deck missing values as 'Unknown' (keep column; missingness itself is informative).")
    else:
        # Default conservative choice: categorical unknown token, numeric median plus missing flag
        if pd.api.types.is_numeric_dtype(eda[col]):
            eda[f"{col}_missing"] = eda[col].isna().astype(int)
            eda[col] = eda[col].fillna(eda[col].median())
            print(f"High-missing numeric {col}: median impute + missing indicator.")
        else:
            eda[col] = eda[col].astype("object").fillna("Unknown")
            print(f"High-missing categorical {col}: encoded as 'Unknown'.")

print("Remaining missing after handling:", int(eda.isna().sum().sum()))
print("EDA frame shape:", eda.shape)
"""))
    cells.append(md("## 2. Univariate analysis: age and fare\n\nHistograms, boxplots, IQR outlier counts, and fare skewness from mean / median / mode."))
    cells.append(code("""uni = pd.read_csv(CSV_PATH)

def iqr_outlier_count(series: pd.Series) -> dict:
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n = int(((s < lower) | (s > upper)).sum())
    return {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper, "n_outliers": n}

age_stats = iqr_outlier_count(uni["age"])
fare_stats = iqr_outlier_count(uni["fare"])
print("Age IQR outlier summary:", age_stats)
print("Fare IQR outlier summary:", fare_stats)

fare_mean = uni["fare"].mean()
fare_median = uni["fare"].median()
fare_mode = uni["fare"].mode().iloc[0]
print(f"fare mean={fare_mean:.4f}")
print(f"fare median={fare_median:.4f}")
print(f"fare mode={fare_mode:.4f}")

if fare_mean > fare_median > fare_mode:
    skew_note = "right-skewed (mean > median > mode)"
elif fare_mean < fare_median < fare_mode:
    skew_note = "left-skewed (mean < median < mode)"
else:
    skew_note = "not a clean mean-median-mode ordering; inspect the histogram"
print("Fare skewness based on mean/median/mode:", skew_note)
"""))
    cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(uni["age"].dropna(), kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Age histogram")
sns.boxplot(x=uni["age"], ax=axes[0, 1], color="steelblue")
axes[0, 1].set_title("Age boxplot")
sns.histplot(uni["fare"].dropna(), kde=True, ax=axes[1, 0], color="darkorange")
axes[1, 0].set_title("Fare histogram")
sns.boxplot(x=uni["fare"], ax=axes[1, 1], color="darkorange")
axes[1, 1].set_title("Fare boxplot")
fig.tight_layout()
fig.savefig(CHARTS / "univariate_age_fare.png", dpi=120)
plt.show()
"""))
    cells.append(md("## 3. Bivariate analysis: survival by sex, pclass, and both\n\nBoolean masking is used where it makes the filter explicit."))
    cells.append(code("""bi = pd.read_csv(CSV_PATH)

survived = bi["survived"] == 1
female = bi["sex"] == "female"
male = bi["sex"] == "male"

print("Survival rate overall:", survived.mean())
print("Survival rate female (mask):", bi.loc[female, "survived"].mean())
print("Survival rate male (mask):", bi.loc[male, "survived"].mean())

print("\\nSurvival by sex:")
print(bi.groupby("sex")["survived"].mean())

print("\\nSurvival by pclass:")
print(bi.groupby("pclass")["survived"].mean())

print("\\nSurvival by sex + pclass:")
print(bi.groupby(["sex", "pclass"])["survived"].mean())

print("\\nMasked example — first-class women:")
first_class_women = female & (bi["pclass"] == 1)
print("n =", int(first_class_women.sum()), "survival =", bi.loc[first_class_women, "survived"].mean())
"""))
    cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sns.barplot(data=bi, x="sex", y="survived", ax=axes[0], errorbar=None)
axes[0].set_title("Survival rate by sex")
sns.barplot(data=bi, x="pclass", y="survived", ax=axes[1], errorbar=None)
axes[1].set_title("Survival rate by pclass")
sns.barplot(data=bi, x="pclass", y="survived", hue="sex", ax=axes[2], errorbar=None)
axes[2].set_title("Survival rate by sex and pclass")
fig.tight_layout()
fig.savefig(CHARTS / "bivariate_survival.png", dpi=120)
plt.show()
"""))
    cells.append(md("## 4. Correlation\n\nExactly these six columns: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`.\nDo **not** include `adult_male` or `alone`."))
    cells.append(code("""corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
assert "adult_male" not in corr_cols
assert "alone" not in corr_cols
corr_df = pd.read_csv(CSV_PATH)[corr_cols]
corr_matrix = corr_df.corr(numeric_only=True)
print(corr_matrix)

abs_off = corr_matrix.abs().copy()
values = np.array(abs_off.values, copy=True)
np.fill_diagonal(values, 0)
abs_off = pd.DataFrame(values, index=abs_off.index, columns=abs_off.columns)
flat = (
    abs_off.where(np.triu(np.ones(abs_off.shape), k=1).astype(bool))
    .stack()
    .sort_values(ascending=False)
)
print("\\nTop two absolute off-diagonal correlations:")
print(flat.head(2))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
ax.set_title("Correlation matrix (six specified columns)")
fig.tight_layout()
fig.savefig(CHARTS / "correlation_heatmap.png", dpi=120)
plt.show()
"""))
    cells.append(md("""### Interpretation of the top two |correlations|

The printed pair above is computed from the actual matrix. Typical Titanic structure:

1. **pclass vs fare** is usually the strongest (negative): higher class number means cheaper tickets, so ticket class and fare move together.
2. The second pair is often **sibsp vs parch** (positive): passengers travelling with siblings/spouses also tend to travel with parents/children, i.e. family-group size.

These are associations, not causal claims. `adult_male` and `alone` were excluded as required.
"""))
    cells.append(md("## 5. Multivariate analysis\n\nFour distinct charts, each with a written interpretation."))
    cells.append(code("""mv = pd.read_csv(CSV_PATH)

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=mv, x="age", y="fare", hue="survived", style="sex", ax=ax)
ax.set_title("Chart 1: Age vs fare by survival and sex")
fig.tight_layout()
fig.savefig(CHARTS / "multivariate_01_scatter_age_fare.png", dpi=120)
plt.show()
"""))
    cells.append(md("""**Chart 1 interpretation.** Fare stretches far above the bulk of passengers, so expensive tickets are visually obvious as a thin cloud at the top. Survival (hue) concentrates among higher fares and among females (style), which matches the bivariate tables. Age alone does not separate survivors as cleanly as fare and sex together. A few elderly passengers with modest fares show that age is not a substitute for class/sex.
"""))
    cells.append(code("""fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=mv, x="pclass", y="age", hue="survived", ax=ax)
ax.set_title("Chart 2: Age by class and survival")
fig.tight_layout()
fig.savefig(CHARTS / "multivariate_02_box_age_class.png", dpi=120)
plt.show()
"""))
    cells.append(md("""**Chart 2 interpretation.** First-class passengers are older on average than third-class passengers, which is a known Titanic pattern (wealth and age travel together). Within a class, the survival split is not a simple age cutoff; children appear more often in the surviving side of third class. The boxes overlap a lot, so age is a supporting variable rather than a standalone rule. Outliers in first class show a long upper tail of older travellers.
"""))
    cells.append(code("""fig = sns.catplot(
    data=mv, x="embarked", y="fare", hue="survived", col="pclass",
    kind="bar", errorbar=None, height=4, aspect=0.8,
)
fig.fig.suptitle("Chart 3: Mean fare by embarkation, survival, and class", y=1.03)
fig.savefig(CHARTS / "multivariate_03_fare_embarked_class.png", dpi=120)
plt.show()
"""))
    cells.append(md("""**Chart 3 interpretation.** Fare still tracks class even after splitting by embarkation port, which confirms that `pclass` is the dominant price driver. Cherbourg (`C`) often shows higher average fares inside first class, consistent with a wealthier boarding group. Survival shading inside each bar group is mixed, so port is a weak extra signal compared with class. Missing embarkation was already rare and handled in profiling.
"""))
    cells.append(code("""fig = sns.pairplot(
    mv[["survived", "pclass", "age", "fare", "sex"]].dropna(),
    hue="survived",
    corner=True,
    diag_kind="hist",
)
fig.fig.suptitle("Chart 4: Pairplot of key numeric fields by survival", y=1.02)
fig.savefig(CHARTS / "multivariate_04_pairplot.png", dpi=120)
plt.show()
"""))
    cells.append(md("""**Chart 4 interpretation.** The pairplot compresses several bivariate views: fare vs pclass is the sharpest numeric pattern, age is diffuse, and survival hue clusters toward lower pclass numbers (i.e. first/second class) and higher fares. Diagonal histograms show that non-survivors dominate the sample, which is why class imbalance is handled later in modeling. This chart is descriptive only; it does not replace a proper train/test split.
"""))
    cells.append(md("""## 6. Standardization demonstration (EDA only)

Z-score `age` and `fare` **for demonstration**. This transformed copy is **not** saved into the modeling CSV and must not leak into `02_modeling.ipynb`. Modeling will fit `StandardScaler` on the training fold only.
"""))
    cells.append(code("""std_src = pd.read_csv(CSV_PATH)
age_before_mean = std_src["age"].mean()
age_before_std = std_src["age"].std()
fare_before_mean = std_src["fare"].mean()
fare_before_std = std_src["fare"].std()

age_z = (std_src["age"] - age_before_mean) / age_before_std
fare_z = (std_src["fare"] - fare_before_mean) / fare_before_std

print("age mean before:", age_before_mean)
print("age std before:", age_before_std)
print("age mean after:", age_z.mean())
print("age std after:", age_z.std())
print("fare mean before:", fare_before_mean)
print("fare std before:", fare_before_std)
print("fare mean after:", fare_z.mean())
print("fare std after:", fare_z.std())
print("NOTE: demonstration only — modeling notebook reloads the original CSV.")
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
    cells.append(md("# Titanic modeling\n\nClassification of `survived` and a separate regression for `fare`.\n\nThis notebook **only** loads `analytics/titanic.csv` created in `01_eda.ipynb`. It does not call `sns.load_dataset`."))
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
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

BASE = Path(".").resolve()
if not (BASE / "02_modeling.ipynb").exists() and (BASE / "analytics" / "02_modeling.ipynb").exists():
    BASE = BASE / "analytics"
elif (BASE / "analytics").exists() and not (BASE / "02_modeling.ipynb").exists():
    BASE = BASE / "analytics"

CSV_PATH = BASE / "titanic.csv"
CHARTS = BASE / "outputs" / "charts"
METRICS = BASE / "outputs" / "metrics"
MODELS = BASE / "models"
CHARTS.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print("Loaded", CSV_PATH, "shape", df.shape)
df.head()
"""))
    cells.append(md("## 1. Train/test split first\n\nStratified split on `survived` **before** any preprocessing. The same split is reused for every classifier."))
    cells.append(code("""feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target_col = "survived"
model_df = df[feature_cols + [target_col]].copy()

X = model_df[feature_cols]
y = model_df[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
print("Train size", X_train.shape, "Test size", X_test.shape)
print("Train survival rate", y_train.mean(), "Test survival rate", y_test.mean())
"""))
    cells.append(md("## 2. Preprocessing pipeline\n\n`ColumnTransformer` + `Pipeline`. Fit **only** on training data."))
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
preprocess.fit(X_train)
X_train_p = preprocess.transform(X_train)
X_test_p = preprocess.transform(X_test)
print("Fitted preprocessor on TRAIN only. Transformed shapes:", X_train_p.shape, X_test_p.shape)

def get_feature_names(fitted_preprocess):
    return fitted_preprocess.get_feature_names_out()
"""))
    cells.append(md("## 3–4. Classifiers and evaluation\n\nLogistic Regression, Decision Tree, Random Forest. Confusion matrix, accuracy, precision, recall, F1, ROC curve, ROC-AUC."))
    cells.append(code("""classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=RANDOM_STATE
    ),
}

rows = []
proba_store = {}
pred_store = {}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, classifiers.items()):
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_score = pipe.predict_proba(X_test)[:, 1]
    pred_store[name] = y_pred
    proba_store[name] = y_score
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    rows.append(metrics)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{name}\\nconfusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    print(name, metrics)

comparison = pd.DataFrame(rows)
comparison.to_csv(METRICS / "classification_comparison.csv", index=False)
print(comparison)
fig.tight_layout()
fig.savefig(CHARTS / "confusion_matrices.png", dpi=120)
plt.show()
"""))
    cells.append(code("""fig, ax = plt.subplots(figsize=(7, 6))
for name, y_score in proba_store.items():
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves (same train/test split)")
ax.legend()
fig.tight_layout()
fig.savefig(CHARTS / "roc_curves.png", dpi=120)
plt.show()
"""))
    cells.append(md("## 5. Decision tree visualization"))
    cells.append(code("""tree_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
])
tree_pipe.fit(X_train, y_train)
feature_names = tree_pipe.named_steps["preprocess"].get_feature_names_out()
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    tree_pipe.named_steps["model"],
    feature_names=feature_names,
    class_names=["died", "survived"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
)
ax.set_title("Decision tree (depth=4) with feature names and class names")
fig.tight_layout()
fig.savefig(CHARTS / "decision_tree.png", dpi=140)
plt.show()
"""))
    cells.append(md("""## 6. Class imbalance

Compare A) baseline logistic regression, B) `class_weight='balanced'`, C) SMOTE on **training data only**.
Never apply SMOTE to the test set.
"""))
    cells.append(code("""def eval_preds(name, y_pred):
    return {
        "method": name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

baseline = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
baseline.fit(X_train, y_train)
base_pred = baseline.predict(X_test)

balanced = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
])
balanced.fit(X_train, y_train)
bal_pred = balanced.predict(X_test)

smote_pipe = ImbPipeline([
    ("preprocess", preprocess),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
smote_pipe.fit(X_train, y_train)
smote_pred = smote_pipe.predict(X_test)

imbalance_table = pd.DataFrame([
    eval_preds("A_baseline", base_pred),
    eval_preds("B_class_weight_balanced", bal_pred),
    eval_preds("C_SMOTE_train_only", smote_pred),
])
imbalance_table.to_csv(METRICS / "imbalance_comparison.csv", index=False)
print(imbalance_table)
print("SMOTE was fit inside the training pipeline only; X_test was never resampled.")
"""))
    cells.append(md("""**Which imbalance method is preferable?** `class_weight='balanced'` is usually the first choice here: it needs no synthetic passengers, keeps the original training distribution of features, and typically lifts recall for the minority survival class without a large precision collapse. SMOTE can help when the minority class is extremely small, but it invents feature combinations that never existed (risky with mixed numeric/one-hot data). The baseline ignores imbalance and often under-recalls survivors. Prefer **B** unless the comparison table shows a clearly better F1 for SMOTE with acceptable precision.
"""))
    cells.append(md("## 7. Random Forest GridSearchCV and OOB score"))
    cells.append(code("""rf_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [4, 6, 8],
    "model__max_features": ["sqrt", "log2"],
}
grid = GridSearchCV(
    rf_pipe,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

best_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
oob_model = RandomForestClassifier(
    oob_score=True,
    random_state=RANDOM_STATE,
    bootstrap=True,
    **best_params,
)
oob_pipe = Pipeline([("preprocess", preprocess), ("model", oob_model)])
oob_pipe.fit(X_train, y_train)
oob_score = oob_pipe.named_steps["model"].oob_score_
print("OOB score:", oob_score)

grid_report = {
    "best_params": grid.best_params_,
    "best_cv_f1": float(grid.best_score_),
    "oob_score": float(oob_score),
}
(METRICS / "random_forest_gridsearch.json").write_text(json.dumps(grid_report, indent=2), encoding="utf-8")
"""))
    cells.append(md("## 8. Regression: predict fare\n\nA separate task. Report MAE, RMSE, R², adjusted R², residual plot, and heteroscedasticity."))
    cells.append(code("""reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "survived"]
reg_df = df[reg_features + ["fare"]].copy()
Xr = reg_df[reg_features]
yr = reg_df["fare"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.25, random_state=RANDOM_STATE
)

reg_num = ["age", "sibsp", "parch", "survived"]
reg_cat = ["pclass", "sex", "embarked"]
reg_preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), reg_num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), reg_cat),
])
reg_pipe = Pipeline([
    ("preprocess", reg_preprocess),
    ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = mean_squared_error(yr_test, yr_pred) ** 0.5
r2 = r2_score(yr_test, yr_pred)
n = len(yr_test)
p = Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
reg_metrics = pd.DataFrame([{
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted_R2": adj_r2,
    "n_test": n,
    "p_predictors": p,
}])
reg_metrics.to_csv(METRICS / "regression_fare.csv", index=False)
print(reg_metrics)

residuals = yr_test.to_numpy() - yr_pred
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(yr_pred, residuals, alpha=0.7)
ax.axhline(0, color="black", linestyle="--")
ax.set_xlabel("Predicted fare")
ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Fare regression residual plot")
fig.tight_layout()
fig.savefig(CHARTS / "regression_residuals.png", dpi=120)
plt.show()

# Heteroscedasticity: residual spread vs predicted level
spread_low = np.std(residuals[yr_pred <= np.median(yr_pred)])
spread_high = np.std(residuals[yr_pred > np.median(yr_pred)])
print("Residual std (low predicted fares):", spread_low)
print("Residual std (high predicted fares):", spread_high)
if spread_high > 1.5 * spread_low:
    hetero_note = (
        "There is evidence of heteroscedasticity: residuals fan out at higher predicted fares, "
        "which is expected because a few luxury tickets have very large fares."
    )
else:
    hetero_note = (
        "Residual spread is relatively stable across predicted fares, so strong heteroscedasticity "
        "is not obvious in this split. A few high-fare points may still stand out visually."
    )
print(hetero_note)
(METRICS / "heteroscedasticity_note.txt").write_text(hetero_note, encoding="utf-8")
"""))
    cells.append(md("## 9. Final comparison and recommendation\n\nClassification metrics and regression metrics stay in separate tables."))
    cells.append(code("""print("CLASSIFICATION (survived)")
print(pd.read_csv(METRICS / "classification_comparison.csv"))
print("\\nIMBALANCE METHODS")
print(pd.read_csv(METRICS / "imbalance_comparison.csv"))
print("\\nREGRESSION (fare) — separate task, do not mix with classification scores")
print(pd.read_csv(METRICS / "regression_fare.csv"))

cls = pd.read_csv(METRICS / "classification_comparison.csv")
best_cls = cls.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]
recommendation = (
    f"The preferred classifier on this fixed stratified split is {best_cls['model']} "
    f"because it has the strongest F1 ({best_cls['f1']:.3f}) with ROC-AUC {best_cls['roc_auc']:.3f}. "
    "Accuracy alone is a weak guide on Titanic because non-survivors are the majority. "
    "A tree ensemble usually captures non-linear class/sex interactions better than a single logistic hyperplane, "
    "while a shallow decision tree is more interpretable but typically less stable. "
    "Use class_weight='balanced' (or inspect the imbalance table) if the operational goal is catching more survivors."
)
print(recommendation)
(METRICS / "final_recommendation.txt").write_text(recommendation, encoding="utf-8")
"""))
    cells.append(md("## 10. Save the complete pipeline and reload raw-input predictions"))
    cells.append(code("""best_name = best_cls["model"]
final_estimator = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=int(best_params.get("n_estimators", 200)),
        max_depth=best_params.get("max_depth", 6),
        max_features=best_params.get("max_features", "sqrt"),
        random_state=RANDOM_STATE,
    ),
}[best_name]

best_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", final_estimator),
])
best_pipeline.fit(X_train, y_train)

model_path = MODELS / "best_pipeline.joblib"
joblib.dump(best_pipeline, model_path)
print("Saved complete preprocessing + estimator pipeline to", model_path)

reloaded = joblib.load(model_path)
raw_input = pd.DataFrame([
    {"pclass": 3, "sex": "male", "age": 22, "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S"},
    {"pclass": 1, "sex": "female", "age": 38, "sibsp": 1, "parch": 0, "fare": 71.2833, "embarked": "C"},
])
raw_pred = reloaded.predict(raw_input)
raw_proba = reloaded.predict_proba(raw_input)
print("Raw unprocessed input:")
print(raw_input)
print("Predictions:", raw_pred)
print("Probabilities:\\n", raw_proba)
assert raw_pred.shape[0] == 2
print("Reload demonstration succeeded: pipeline accepted RAW/unprocessed rows.")
"""))
    nb["cells"] = cells
    nbf.write(nb, NB2)
    print("Wrote", NB2)


if __name__ == "__main__":
    write_eda()
    write_modeling()
