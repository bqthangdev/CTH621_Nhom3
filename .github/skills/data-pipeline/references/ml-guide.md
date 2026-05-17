# ML Guide — CTH621

## Common Rules (All Tasks)

- `random_state` MUST come from `params.yaml` — NEVER hardcode
- Each model class MUST accept `**kwargs` and pass them to the underlying estimator
- All metric results MUST be appended to `summary_results.csv`
- Save model as `.joblib` to `outputs/{dataset}/models/`
- Before training, check `progress.json` — skip if step is "DONE"

---

## 4.1 Classification (Dataset Group A)

### Requirements
- At least **3 different target columns** experimented sequentially
- Train/Test split: **80/20** (use `stratify=y` when possible)
- At least **2 algorithms** (e.g. Logistic Regression + Decision Tree + SVM)

### Evaluation Output (per target, per algorithm)
```python
# Must print/log ALL of:
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_score(y_test, y_pred, average='weighted', zero_division=0)
# Also log class distribution to detect imbalance:
logger.info(f"Class distribution: {y.value_counts(normalize=True).to_dict()}")
```

### Code Pattern
```python
class ClassificationPipeline:
    def __init__(self, algorithm: str, config: dict):
        algo_map = {"logistic": LogisticRegression, "decision_tree": DecisionTreeClassifier, "svm": SVC}
        self.model = algo_map[algorithm](random_state=config["random_state"], **config.get("kwargs", {}))
```

---

## 4.2 Regression (Dataset Group B — Time Series)

### CRITICAL: NO Random Split
```python
# CORRECT — time-aware split:
split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

# FORBIDDEN:
# train_test_split(X, y, random_state=...)  ← DO NOT USE
```

### Algorithms
- Linear Regression (baseline)
- ARIMA / SARIMA (via `statsmodels` — use `order` and `seasonal_order` from params.yaml)
- XGBoost Regressor (with rolling-window features: lag_1, lag_7, rolling_mean_7)

### Evaluation
```python
mean_absolute_error(y_test, y_pred)    # MAE
np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
r2_score(y_test, y_pred)               # R²
```

### Rolling-Window Feature Engineering (for XGBoost on time series)
```python
for lag in config["regression"]["lags"]:       # e.g. [1, 7, 14]
    df[f"lag_{lag}"] = df[target].shift(lag)
for w in config["regression"]["rolling_windows"]:  # e.g. [7, 30]
    df[f"rolling_mean_{w}"] = df[target].rolling(w).mean()
df.dropna(inplace=True)
```

---

## 4.3 Clustering (All Groups A, B, C — independently)

### Pre-processing
```python
# MUST drop all label/target columns before clustering:
feature_cols = [c for c in df.columns if c not in config["label_columns"]]
X = df[feature_cols]
```

### Three Algorithms (all three required)

#### K-Means
```python
# Find optimal K: Elbow Method + Silhouette Score
inertias, silhouettes = [], []
k_range = range(2, config["clustering"]["max_k"] + 1)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=config["random_state"], **kwargs)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
# Plot Elbow curve → save PNG
# Use best_k = argmax(silhouettes) + 2
```

#### Hierarchical Clustering
```python
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X_scaled, method=config["clustering"]["linkage_method"])  # e.g. "ward"
# Plot Dendrogram → save PNG
# Cut tree at config["clustering"]["n_clusters"]
```

#### DBSCAN (Outlier Isolation)
```python
db = DBSCAN(eps=config["clustering"]["dbscan_eps"],
            min_samples=config["clustering"]["dbscan_min_samples"])
labels = db.fit_predict(X_scaled)
n_outliers = (labels == -1).sum()
logger.info(f"DBSCAN: {n_outliers} outliers detected ({n_outliers/len(labels)*100:.1f}%)")
```

### Clustering Output Files
```
outputs/{dataset}/ml/clustering/
├── elbow_curve.png
├── silhouette_scores.png
├── dendrogram.png
├── kmeans_labels.csv
├── hierarchical_labels.csv
└── dbscan_labels.csv       # cluster=-1 are outliers
```
