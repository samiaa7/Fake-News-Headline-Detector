import pandas as pd
from src.models import Detector
from src.explain import explain
from src.data import HeadlineDataset

df = pd.read_csv('data/headlines_clean.csv')
print(df.columns.tolist())

sample = df.sample(10, random_state=1)
headlines = sample['title'].tolist()

# add 2 of your own typed headlines (these need cleaning)
own = [
    "BREAKING: Obama secretly plans to cancel the election",
    "Federal Reserve raises interest rates by quarter point",
]

for name in ['logistic', 'naive_bayes']:
    det = Detector.load(f'models/{name}.joblib')
    print(f"\n===== {name} =====")
    for h in headlines + own:
        cleaned = h if h in headlines else HeadlineDataset._clean_title(h)
        proba = det.predict_proba([cleaned])[0]
        label = det.classes_[proba.argmax()]
        print(f"\n{h}\n  -> {label.upper()} ({proba.max():.0%})")
        for word, score in explain(det, cleaned, top_k=5):
            side = 'fake' if score > 0 else 'real'
            print(f"     {word}: {side} ({score:+.2f})")