from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


class TopicClusterer:
    def __init__(self, seed=42):
        self.seed = seed
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.kmeans = None
        self.k = None
        self.scores = {}

    def choose_k(self, X, k_range=range(4, 13)):
        scores = {}
        for k in k_range:
            labels = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit_predict(X)
            scores[k] = silhouette_score(X, labels, sample_size=5000, random_state=self.seed)
        return max(scores, key=scores.get), scores

    def fit(self, texts, k=None):
        X = self.vectorizer.fit_transform(texts)
        if k is None:
            k, self.scores = self.choose_k(X)
        self.k = k
        self.kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(X)
        return self

    def top_terms(self, n=8):
        terms = self.vectorizer.get_feature_names_out()
        order = self.kmeans.cluster_centers_.argsort(axis=1)[:, ::-1][:, :n]
        return {c: [terms[i] for i in order[c]] for c in range(self.k)}

    def predict(self, texts):
        return self.kmeans.predict(self.vectorizer.transform(texts))
