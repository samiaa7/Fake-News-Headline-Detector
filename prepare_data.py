import os
from src.data import HeadlineDataset

ds = HeadlineDataset("data/raw/Fake.csv", "data/raw/True.csv").load_csv()
print("raw rows:", len(ds.df))

ds.clean()
print("clean rows:", len(ds.df))
print(ds.df["label"].value_counts(normalize=True).round(3))

ds.df.to_csv("data/headlines_clean.csv", index=False)

train, test = ds.split(seed=42)
os.makedirs("data/split", exist_ok=True)
train.to_csv("data/split/train.csv", index=False)
test.to_csv("data/split/test.csv", index=False)

print("train:", len(train), "test:", len(test))