import os
import json
import joblib
import streamlit as st
from src.models import Detector
from src.explain import explain
from src.data import HeadlineDataset
from src.clusters import TopicClusterer  # needed so joblib can load the clusterer

# ---------- page setup ----------
st.set_page_config(page_title='Fake News Headline Detector')
st.title('Fake News Headline Detector')


# ---------- load things once (cached so the app stays fast) ----------
@st.cache_resource
def load_models():
    return {
        'Logistic regression': Detector.load('models/logistic.joblib'),
        'Naive Bayes': Detector.load('models/naive_bayes.joblib'),
    }


@st.cache_resource
def load_clusterer():
    # returns (clusterer, names) or (None, None) if files are missing
    if not os.path.exists('models/clusterer.joblib'):
        return None, None
    clusterer = joblib.load('models/clusterer.joblib')
    with open('models/cluster_names.json') as f:
        names = json.load(f)  # keys are strings like "0", "1", ...
    return clusterer, names


# helper: show an image if it exists, otherwise a friendly message
def show(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption)
    else:
        st.info(f'{path} is not available yet.')


models = load_models()
clusterer, cluster_names = load_clusterer()

# ---------- tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(['Detector', 'EDA', 'Model comparison', 'Topics'])

# ----- Tab 1: the detector -----
with tab1:
    choice = st.selectbox('Model', list(models))
    headline = st.text_area('Paste a headline')

    if st.button('Check') and headline.strip():
        det = models[choice]

        # clean the headline first, because the models were trained on cleaned text
        cleaned = HeadlineDataset._clean_title(headline)

        # prediction: column 0 = fake, column 1 = real
        proba = det.predict_proba([cleaned])[0]
        label = det.classes_[proba.argmax()]
        st.subheader(f'{label.upper()} ({proba.max():.0%} confident)')

        # explanation: positive score = pushes toward fake, negative = toward real
        st.write('Words that influenced the result:')
        for word, score in explain(det, cleaned, top_k=5):
            side = 'fake' if score > 0 else 'real'
            st.write(f'{word}: pushes toward {side} ({score:+.2f})')

        # topic of the headline (extra)
        if clusterer is not None:
            cluster_id = int(clusterer.predict([cleaned])[0])
            st.write(f'Topic: **{cluster_names[str(cluster_id)]}**')

# ----- Tab 2: EDA charts -----
with tab2:
    show('figures/eda_length.png', 'Headline length by label')
    show('figures/eda_top_words.png', 'Top words per class')
    show('figures/eda_subject.png', 'Subject by label')

# ----- Tab 3: model comparison charts -----
with tab3:
    show('figures/confusion_matrices.png', 'Confusion matrices')
    show('figures/model_comparison.png', 'Model comparison')

# ----- Tab 4: topic clusters -----
with tab4:
    show('figures/clusters.png', 'Headline topic clusters')
