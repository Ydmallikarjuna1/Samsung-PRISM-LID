import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path("datasets/voxlingua107_wav")

metadata = pd.read_csv(ROOT / "metadata.csv")

# 80% train, 20% temp
train_df, temp_df = train_test_split(
    metadata,
    test_size=0.20,
    stratify=metadata["label"],
    random_state=42,
)

# Split remaining 20% into validation and test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42,
)

train_df.to_csv(ROOT / "train.csv", index=False)
val_df.to_csv(ROOT / "val.csv", index=False)
test_df.to_csv(ROOT / "test.csv", index=False)

print("=" * 60)
print("Dataset Split Complete")
print("=" * 60)

print(f"Train Samples : {len(train_df)}")
print(f"Validation    : {len(val_df)}")
print(f"Test          : {len(test_df)}")

print("\nTraining Distribution")
print(train_df["language"].value_counts())

print("\nValidation Distribution")
print(val_df["language"].value_counts())

print("\nTest Distribution")
print(test_df["language"].value_counts())