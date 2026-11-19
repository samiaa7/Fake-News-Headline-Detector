import numpy as np


def explain(detector, headline, top_k=5):
    tfidf = detector.pipeline.named_steps['tfidf']
    clf = detector.pipeline.named_steps['clf']
    x = tfidf.transform([headline])
    words = tfidf.get_feature_names_out()
    idx = x.nonzero()[1]
    if len(idx) == 0:
        return []
    values = x.toarray()[0][idx]

    if hasattr(clf, 'coef_'):
        # logistic regression: coef_ is for classes_[1] = real, so flip the sign to score fake
        weights = -clf.coef_[0][idx]
    elif hasattr(clf, 'feature_log_prob_'):
        # naive bayes: log P(word | fake) - log P(word | real)
        weights = clf.feature_log_prob_[0][idx] - clf.feature_log_prob_[1][idx]
    else:
        return []

    scores = weights * values
    order = np.argsort(-np.abs(scores))[:top_k]
    return [(words[idx[i]], float(scores[i])) for i in order]
