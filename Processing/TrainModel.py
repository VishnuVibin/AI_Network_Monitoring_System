import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "../data/processed/ml_features_v3.csv"
MODEL_FILE = "../data/processed/network_anomaly_model_v3.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Dataset V3...")

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\n0 = Normal")
print("1 = Anomaly")


# ============================================================
# PREPARE FEATURES
# ============================================================

# Remove columns that should not be used directly for ML
columns_to_remove = [
    "label",
    "timestamp",
    "window"
]

columns_to_remove = [
    column for column in columns_to_remove
    if column in df.columns
]

X = df.drop(columns=columns_to_remove)

y = df["label"]


# ============================================================
# HANDLE INVALID VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# ============================================================
# DISPLAY FEATURES
# ============================================================

print("\nFeatures used for training:")

for feature in X.columns:
    print("-", feature)

print("\nNumber of features:", X.shape[1])


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset split:")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DATASET V3 MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Anomaly"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Normal",
        "Anomaly"
    ],
    yticklabels=[
        "Normal",
        "Anomaly"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Network Anomaly Detection - Dataset V3"
)

plt.tight_layout()

plt.show()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "../data/processed/feature_importance_v3.csv",
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\nModel saved successfully!")

print(
    "Model:",
    MODEL_FILE
)


# ============================================================
# FINISHED
# ============================================================

print("\nTraining process completed!")