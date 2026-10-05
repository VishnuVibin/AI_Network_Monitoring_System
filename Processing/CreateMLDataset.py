import pandas as pd

NORMAL_FILE = "../data/processed/normal_loopback_10min_features.csv"
ANOMALY_FILE = "../data/processed/anomaly_loopback_10min_features.csv"

OUTPUT_FILE = "../data/processed/ml_features_v3.csv"

print("Loading normal features...")
normal = pd.read_csv(NORMAL_FILE)

print("Normal windows:", len(normal))

print("\nLoading anomaly features...")
anomaly = pd.read_csv(ANOMALY_FILE)

print("Anomaly windows:", len(anomaly))

# Labels
normal["label"] = 0
anomaly["label"] = 1

# Combine
data = pd.concat(
    [normal, anomaly],
    ignore_index=True
)

# Shuffle
data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 50)
print("DATASET V3 CREATED")
print("=" * 50)

print("Total samples:", len(data))

print("\nLabel distribution:")
print(data["label"].value_counts())

print("\nExpected:")
print("0 = Normal")
print("1 = Anomaly")

print("\nShape:", data.shape)

print("\nSaved to:")
print(OUTPUT_FILE)