import numpy as np
from src.models import BaselineDetector, NaiveBayesDetector, LogisticDetector, Detector

TEXTS = ['trump slams critics in wild speech', 'senate passes budget bill after debate'] * 20
LABELS = ['fake', 'real'] * 20


def test_proba_shape_and_sum():
    for cls in [NaiveBayesDetector, LogisticDetector, BaselineDetector]:
        det = cls().fit(TEXTS, LABELS)
        proba = det.predict_proba(TEXTS)
        assert proba.shape == (len(TEXTS), 2)
        assert np.allclose(proba.sum(axis=1), 1)


def test_classes_order():
    det = LogisticDetector().fit(TEXTS, LABELS)
    assert list(det.classes_) == ['fake', 'real']


def test_save_load_roundtrip(tmp_path):
    det = NaiveBayesDetector().fit(TEXTS, LABELS)
    path = tmp_path / 'm.joblib'
    det.save(path)
    loaded = Detector.load(path)
    assert list(loaded.predict(TEXTS)) == list(det.predict(TEXTS))


def test_baseline_predicts_majority():
    labels = ['fake'] * 30 + ['real'] * 10
    det = BaselineDetector().fit(['some headline here'] * 40, labels)
    assert set(det.predict(['anything at all'])) == {'fake'}

