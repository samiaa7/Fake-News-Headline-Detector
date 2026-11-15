
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


def make_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), max_features=20000)

class Detector:
    name = 'detector'

    def __init__(self):
        self.pipeline = None

    def build(self):
        raise NotImplementedError

    def fit(self, texts, labels):
        self.pipeline = self.build()
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)

    @property
    def classes_(self):
        return self.pipeline.classes_

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class BaselineDetector(Detector):
    name = 'baseline'

    def build(self):
        return Pipeline([('tfidf', make_tfidf()),
                         ('clf', DummyClassifier(strategy='most_frequent'))])


class NaiveBayesDetector(Detector):
    name = 'naive_bayes'

    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

    def build(self):
        return Pipeline([('tfidf', make_tfidf()),
                         ('clf', MultinomialNB(alpha=self.alpha))])


class LogisticDetector(Detector):
    name = 'logistic'

    def __init__(self, C=10):
        super().__init__()
        self.C = C

    def build(self):
        return Pipeline([('tfidf', make_tfidf()), ('clf', LogisticRegression(C=self.C, max_iter=1000,random_state=42))])

