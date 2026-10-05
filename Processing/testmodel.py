import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# FILES
# ============================================================

MODEL_FILE = "../data/processed/network_anomaly_model_v3.pkl"
DATASET_FILE = "../data/processed/ml_features.csv"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading Dataset V3...")

df = pd.read_csv(DATASET_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# PREPARE DATA
# ============================================================

# Save actual labels
y_true = df["label"]

# Remove non-feature columns
columns_to_remove = [
    "label",
    "timestamp",
    "window"
]

columns_to_remove = [
    col for col in columns_to_remove
    if col in df.columns
]

X = df.drop(columns=columns_to_remove)

# Handle invalid values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# ============================================================
# PREDICT
# ============================================================

print("\nRunning predictions...")

predictions = model.predict(X)

probabilities = model.predict_proba(X)


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    y_true,
    predictions
)

print("\n" + "=" * 60)
print("MODEL TEST RESULTS")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_true,
        predictions,
        target_names=[
            "Normal",
            "Anomaly"
        ],
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_true,
        predictions
    )
)


# ============================================================
# DISPLAY INDIVIDUAL PREDICTIONS
# ============================================================

results = df[
    [
        "timestamp",
        "window",
        "label"
    ]
].copy()

results["predicted"] = predictions

results["normal_probability"] = probabilities[:, 0]

results["anomaly_probability"] = probabilities[:, 1]


# Convert labels to readable names

results["actual_status"] = results["label"].map({
    0: "NORMAL",
    1: "ANOMALY"
})

results["predicted_status"] = results["predicted"].map({
    0: "NORMAL",
    1: "ANOMALY"
})


print("\nFirst 20 predictions:")

print(
    results[
        [
            "window",
            "actual_status",
            "predicted_status",
            "normal_probability",
            "anomaly_probability"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# INCORRECT PREDICTIONS
# ============================================================

wrong = results[
    results["label"] != results["predicted"]
]

print("\nIncorrect predictions:", len(wrong))

if len(wrong) > 0:

    print("\nIncorrect samples:")

    print(
        wrong.to_string(index=False)
    )

else:

    print("\nAll samples classified correctly!")


# ============================================================
# SAVE TEST RESULTS
# ============================================================

results.to_csv(
    "../data/processed/model_test_results.csv",
    index=False
)

print("\nTest results saved to:")
print("../data/processed/model_test_results.csv")