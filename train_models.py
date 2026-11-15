import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from src.models import BaselineDetector, NaiveBayesDetector, LogisticDetector

train = pd.read_csv('data/split/train.csv')
test = pd.read_csv('data/split/test.csv')
os.makedirs('models', exist_ok=True)

for cls in [BaselineDetector, NaiveBayesDetector, LogisticDetector]:
    det = cls().fit(train['title'], train['label'])
    pred = det.predict(test['title'])
    acc = accuracy_score(test['label'], pred)
    f1 = f1_score(test['label'], pred, pos_label='fake')
    print(f'{det.name}: accuracy {acc:.4f}, F1 (fake) {f1:.4f}')
    det.save(f'models/{det.name}.joblib')
