# Kaggle - Historical Flight Delay and Weather Data USA (May - December 2019)

# CHECKLIST 1:
# Indicate data source (Kaggle - Historical Flight Delay and Weather Data USA (May - December 2019))
# Load data from multiple CSV files (merge into a single DataFrame)
# Show sample rows
# Provide information about data structure (number of samples, attributes, classes)

import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

files = glob.glob("sample_data/*.csv")
print("Found files:", files)

dataframes = []
for f in files:
    df_temp = pd.read_csv(f)
    print(f"File {f} loaded, shape: {df_temp.shape}")
    dataframes.append(df_temp)

df = pd.concat(dataframes, ignore_index=True)

print("\nData preview (df.head()):")
print(df.head())

print("\nDataFrame info:")
print(df.info())

print("\nData size:", df.shape)

# CHECKLIST 2a:
# Check missing values in each column
# Clean missing values and anomalies:
# - remove actual_arrival_dt and actual_departure_dt (not used for predicting delay)
# - tail_number (aircraft identifier)
# - weather columns (x/y) (median imputation)

missing = df.isnull().sum().sort_values(ascending=False)
print("Missing values in each column:")
print(missing.head(20))

cols_to_drop = [
    'actual_arrival_dt',
    'actual_departure_dt',
    'tail_number'
]

df_clean = df.drop(columns=cols_to_drop)
print("Removed columns:", cols_to_drop)
print("New shape:", df_clean.shape)

weather_cols = [
    'HourlyPrecipitation_x', 'HourlyPrecipitation_y',
    'HourlyStationPressure_x', 'HourlyStationPressure_y',
    'HourlyWindSpeed_x', 'HourlyWindSpeed_y',
    'HourlyVisibility_x', 'HourlyVisibility_y',
    'HourlyDryBulbTemperature_x', 'HourlyDryBulbTemperature_y',
    'STATION_x', 'STATION_y'
]

for col in weather_cols:
    median_val = df_clean[col].median()
    df_clean[col].fillna(median_val, inplace=True)

print("\nMissing values after imputation:")
print(df_clean[weather_cols].isnull().sum())

# CHECKLIST 2b:
# Encode categorical data:
# - carrier_code
# - origin_airport
# - destination_airport
# - cancelled_code (NaN → None)

df_clean['cancelled_code'] = df_clean['cancelled_code'].fillna('None')

categorical_cols = [
    'carrier_code',
    'origin_airport',
    'destination_airport',
    'cancelled_code'
]

df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

print("Shape after categorical encoding:", df_encoded.shape)
print("Example columns after OHE:")
print(df_encoded.columns[:40])

# CHECKLIST 2c:
# Type optimization and downsampling to 300,000 rows
# - convert float64 -> float32, int64 -> int32
# - random sampling to 300,000 records (downsampling) to reduce RAM usage

print("\n>>> INFO BEFORE TYPE OPTIMIZATION <<<")
print(df_encoded.info(memory_usage="deep"))

df_opt = df_encoded.copy()

float_cols = df_opt.select_dtypes(include=["float64"]).columns
int_cols = df_opt.select_dtypes(include=["int64"]).columns

print("\nNumber of float64 columns:", len(float_cols))
print("Number of int64 columns:", len(int_cols))

for col in float_cols:
    df_opt[col] = df_opt[col].astype("float32")

for col in int_cols:
    df_opt[col] = df_opt[col].astype("int32")

print("\n>>> INFO AFTER TYPE OPTIMIZATION <<<")
print(df_opt.info(memory_usage="deep"))

N = 75000

print("\nShape before downsampling:", df_opt.shape)
if len(df_opt) > N:
    df_model = df_opt.sample(n=N, random_state=42)
    print("Random sampling executed to:", df_model.shape)
else:
    df_model = df_opt
    print("Dataset has fewer than 100,000 rows – unchanged:", df_model.shape)

print("\nPreview of df_model.head():")
print(df_model.head())

# CHECKLIST 2d:
# Binary target and scaling
# Split into training and test sets

df_model["delayed"] = (df_model["arrival_delay"] > 15).astype(int)

print("\nUnique classes in y (delayed):", df_model["delayed"].unique())

target_col = "delayed"

cols_to_exclude = [
    "date",
    "scheduled_departure_dt",
    "scheduled_arrival_dt",
    "arrival_delay",
]

cols_to_exclude = [c for c in cols_to_exclude if c in df_model.columns]

feature_cols = [c for c in df_model.columns if c not in cols_to_exclude + [target_col]]

X = df_model[feature_cols]
y = df_model[target_col]

print("\n>>> FEATURE MATRIX / TARGET <<<")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Example X columns:", feature_cols[:10])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset split:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nAfter scaling (numpy arrays):")
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled  shape:", X_test_scaled.shape)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
print("\nPreview X_train_scaled_df.head():")
print(X_train_scaled_df.head())

# CHECKLIST 3a:
# Define models and configuration

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
        solver="lbfgs"
    ),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=50,
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"
    ),
    "SVM_RBF": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
        max_iter=200
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        n_jobs=-1
    ),
    "NaiveBayes": GaussianNB(),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=15,
        random_state=42,
        verbose=False
    )
}

print("\n>>> DEFINED MODELS <<<")
for name, model in models.items():
    print(f"- {name}: {model}")

# CHECKLIST 3b:
# Train models and make predictions

results = []

for name, model in models.items():
    print(f"\n>>> Training model: {name} ...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    else:
        roc = np.nan

    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Balanced accuracy:  {bacc:.4f}")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall:             {rec:.4f}")
    print(f"  F1-score:           {f1:.4f}")
    print(
        f"  ROC AUC:            {roc:.4f}"
        if not np.isnan(roc)
        else "  ROC AUC:   (none, model has no predict_proba)"
    )

    results.append({
        "model": name,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc
    })

# CHECKLIST 3c:
# Model summary

results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by="f1", ascending=False)

print("\n>>> MODEL SUMMARY (sorted by F1) <<<")
print(results_df_sorted)

# CHECKLIST 4a:
print("\n>>> RandomizedSearchCV for GradientBoosting <<<")

X_tune, _, y_tune, _ = train_test_split(
    X_train_scaled,
    y_train,
    train_size=20000,
    stratify=y_train,
    random_state=42
)

gb_base = GradientBoostingClassifier(random_state=42)

param_dist = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4],
    "subsample": [0.7, 0.9, 1.0]
}

gb_search = RandomizedSearchCV(
    estimator=gb_base,
    param_distributions=param_dist,
    scoring="f1",
    n_iter=10,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

gb_search.fit(X_tune, y_tune)

print("\nBest parameters:", gb_search.best_params_)
print("Best F1 score (CV):", gb_search.best_score_)

# CHECKLIST 4b:
# Compare metrics after tuning

best_gb = gb_search.best_estimator_

y_pred_tuned = best_gb.predict(X_test_scaled)
y_proba_tuned = best_gb.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred_tuned)
prec = precision_score(y_test, y_pred_tuned, zero_division=0)
rec = recall_score(y_test, y_pred_tuned, zero_division=0)
f1 = f1_score(y_test, y_pred_tuned, zero_division=0)
auc = roc_auc_score(y_test, y_proba_tuned)

print("\n>>> Results GradientBoosting (tuned) <<<")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

results.append({
    "model": "GradientBoosting_tuned",
    "accuracy": acc,
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_tuned),
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "roc_auc": auc
})

df_results = pd.DataFrame(results).sort_values(by="f1", ascending=False)
print("\n>>> MODEL COMPARISON (after tuning) <<<")
print(df_results)

# CHECKLIST 4c:
# Confusion Matrix and ROC Curve

print("\n>>> Confusion Matrix and ROC Curve for GradientBoosting_tuned <<<")

cm = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No delay", "Delay"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - GradientBoosting (tuned)")
plt.show()

RocCurveDisplay.from_predictions(y_test, y_proba_tuned)
plt.title("ROC Curve - GradientBoosting (tuned)")
plt.show()
