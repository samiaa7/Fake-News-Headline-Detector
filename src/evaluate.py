import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, confusion_matrix, make_scorer)
from sklearn.model_selection import StratifiedKFold, cross_validate as sk_cross_validate

POS = 'fake'
LABELS = ['fake', 'real']
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'log_loss']
COLUMN_NAMES = {'accuracy': 'Accuracy', 'precision': 'Precision (fake)', 'recall': 'Recall (fake)',
                'f1': 'F1 (fake)', 'log_loss': 'Log-loss'}
DISPLAY = {'baseline': 'Baseline', 'naive_bayes': 'Naive Bayes', 'logistic': 'Logistic regression'}


class Evaluator:
    """Scores any Detector (or fitted scikit-learn Pipeline) with the project's metrics.
    The fake class is the positive class for precision, recall and F1."""

    def __init__(self, train, test, text_col='title', label_col='label', seed=42):
        self.X_train, self.y_train = train[text_col], train[label_col]
        self.X_test, self.y_test = test[text_col], test[label_col]
        self.seed = seed
        self.scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, pos_label=POS, zero_division=0),
            'recall': make_scorer(recall_score, pos_label=POS, zero_division=0),
            'f1': make_scorer(f1_score, pos_label=POS, zero_division=0),
            'log_loss': 'neg_log_loss',
        }

    def cross_validate(self, detector, n_splits=5, pipeline=None):
        """5-fold stratified CV on the training set only. The TF-IDF vectorizer is refit
        inside every fold (it is a step of the pipeline), so no vocabulary leaks across folds.
        Returns {metric: (mean, std)}."""
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        pipe = pipeline if pipeline is not None else detector.build()
        res = sk_cross_validate(pipe, self.X_train, self.y_train, cv=cv, scoring=self.scoring)
        out = {}
        for m in METRICS:
            vals = res[f'test_{m}']
            if m == 'log_loss':
                vals = -vals
            out[m] = (float(vals.mean()), float(vals.std()))
        return out

    def test_metrics(self, model):
        """Metrics on the held-out test set. `model` must already be fitted and offer
        predict, predict_proba and classes_ (a Detector or a fitted Pipeline)."""
        pred = model.predict(self.X_test)
        proba = model.predict_proba(self.X_test)
        return {
            'accuracy': accuracy_score(self.y_test, pred),
            'precision': precision_score(self.y_test, pred, pos_label=POS, zero_division=0),
            'recall': recall_score(self.y_test, pred, pos_label=POS, zero_division=0),
            'f1': f1_score(self.y_test, pred, pos_label=POS, zero_division=0),
            'log_loss': log_loss(self.y_test, proba, labels=list(model.classes_)),
            'confusion': confusion_matrix(self.y_test, pred, labels=LABELS),
        }


def make_cv_table(cv_results):
    rows = {DISPLAY.get(name, name): {COLUMN_NAMES[m]: f'{res[m][0]:.4f} ± {res[m][1]:.4f}' for m in METRICS}
            for name, res in cv_results.items()}
    return pd.DataFrame(rows).T


def make_test_table(test_results):
    rows = {DISPLAY.get(name, name): {COLUMN_NAMES[m]: round(float(res[m]), 4) for m in METRICS}
            for name, res in test_results.items()}
    return pd.DataFrame(rows).T


def plot_confusion_matrices(test_results, names=('naive_bayes', 'logistic')):
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        cm = test_results[name]['confusion']
        ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(LABELS)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(LABELS)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=13,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(DISPLAY.get(name, name))
    fig.tight_layout()
    return fig


def plot_model_comparison(test_results, names=('baseline', 'naive_bayes', 'logistic')):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    width = 0.35
    acc = [test_results[n]['accuracy'] for n in names]
    f1 = [test_results[n]['f1'] for n in names]
    bars1 = ax.bar(x - width / 2, acc, width, label='Accuracy', color='#2e86ab')
    bars2 = ax.bar(x + width / 2, f1, width, label='F1 (fake)', color='#d1495b')
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f'{b.get_height():.3f}',
                    ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(n, n) for n in names])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.set_title('Model comparison on the test set')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig
