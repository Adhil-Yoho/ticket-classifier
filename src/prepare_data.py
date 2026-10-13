import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/tickets.csv")
df = df.dropna(subset=["text", "category", "priority"])

df["text"] = df["text"].str.lower().str.strip()
df = df[df["text"] !=""]
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("Dataset shape:", df.shape)
print("\nCategory counts:")
print(df["category"].value_counts().sort_index())
print("\nPriority counts:")
print(df["priority"].value_counts().sort_index())

df["stratify_label"] = (
    df["category"].astype(str) + "_" + df["priority"].astype(str)
)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["stratify_label"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["stratify_label"],
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df = train_df.drop(columns=["stratify_label"])
val_df = val_df.drop(columns=["stratify_label"])
test_df = test_df.drop(columns=["stratify_label"])

train_df.to_csv(
    "data/train.csv", index=False
)
val_df.to_csv(
    "data/validation.csv", index=False
)
test_df.to_csv(
    "data/test.csv", index=False
)

print("\nTraining rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))
