# ===============================
# Streamlit App - Sentiment Analysis
# ===============================

import os
import re
from pathlib import Path

import joblib
import streamlit as st

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import ToktokTokenizer
except ImportError:
    nltk = None

st.set_page_config(
    page_title="ReviewSignal | Product insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "assets" / "styles.css"
if os.path.exists(css_path):
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_path = model_dir / "logreg_sentiment_model_full.pkl"
    vectorizer_path = model_dir / "tfidf_vectorizer_full.pkl"
    missing = [path.name for path in (model_path, vectorizer_path) if not path.exists()]
    if missing:
        return None, None, missing
    return joblib.load(model_path), joblib.load(vectorizer_path), []

model, vectorizer, missing_files = load_model()

if model is None or vectorizer is None:
    st.error("The prediction engine is not available in this deployment.")
    st.markdown(
        "Missing: **" + ", ".join(missing_files) + "**. "
        "Commit the production files in `models/` and redeploy the app."
    )
    st.stop()


@st.cache_resource
def get_text_tools():
    fallback_stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
        "were", "with", "you",
    }
    if nltk is None:
        return fallback_stop_words, None, None
    try:
        words = set(stopwords.words("english"))
        nltk.data.find("corpora/wordnet")
        return words, WordNetLemmatizer(), ToktokTokenizer()
    except LookupError:
        return fallback_stop_words, None, ToktokTokenizer()


stop_words, lemmatizer, tokenizer = get_text_tools()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = tokenizer.tokenize(text) if tokenizer else text.split()
    tokens = [word for word in tokens if word not in stop_words]
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def predict_review(review):
    features = vectorizer.transform([preprocess_text(review)])
    probabilities = model.predict_proba(features)[0]
    scores = {label: float(score) for label, score in zip(model.classes_, probabilities)}
    sentiment = str(model.predict(features)[0]).lower()
    return sentiment, scores

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">REVIEWS / SIGNAL / ACTION</div>
        <h1>ReviewSignal</h1>
        <p>Understand the feeling behind every customer review in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ReviewSignal")
    st.caption("A focused sentiment workspace for product teams.")
    st.markdown("#### Try a sample")
    samples = {
        "Positive review": "The fabric feels premium and the fit is exactly as described. I would buy this again.",
        "Critical review": "The color looked different from the photos and the stitching started to unravel after one wash.",
    }
    selected_sample = st.selectbox("Load an example", ["Choose a sample"] + list(samples), label_visibility="collapsed")
    st.divider()
    st.markdown("#### How it works")
    st.caption("Your review is cleaned, converted into TF-IDF features, and scored by a trained logistic regression model.")
    st.caption("No review text is stored by this app.")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    default_review = samples.get(selected_sample, "")
    review_input = st.text_area(
        "Customer review",
        value=default_review,
        height=210,
        max_chars=2000,
        placeholder="Paste a customer review to reveal its sentiment...",
        help="For the clearest signal, include the customer's specific experience.",
    )
    st.caption(f"{len(review_input):,} / 2,000 characters")
    analyze = st.button("Analyze review", type="primary", use_container_width=True)
    if analyze:
        if not review_input.strip():
            st.warning("Add a review before running the analysis.")
        else:
            prediction, scores = predict_review(review_input)
            positive_score = scores.get("positive", 0.0)
            negative_score = scores.get("negative", 0.0)
            confidence = scores.get(prediction, max(positive_score, negative_score))
            sentiment_label = "Positive" if prediction == "positive" else "Negative"
            color = "#16866a" if prediction == "positive" else "#c34c4c"

            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-label">SENTIMENT RESULT</div>
                  <h2 style="color:{color};">{sentiment_label}</h2>
                  <p class="result-status">The model is <strong>{confidence:.0%}</strong> confident in this signal.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            metric1, metric2 = st.columns(2)
            metric1.metric("Positive signal", f"{positive_score:.0%}")
            metric2.metric("Negative signal", f"{negative_score:.0%}")
            st.progress(confidence, text=f"Model confidence: {confidence:.0%}")

with col2:
    st.markdown('<div class="insight-panel"><div class="panel-eyebrow">PRODUCT TEAM VIEW</div><h2>From words to direction.</h2><p>Use sentiment as an early signal for what customers love, what frustrates them, and where the product experience needs attention.</p></div>', unsafe_allow_html=True)
    st.markdown("#### Built for a quick read")
    st.markdown('<div class="feature-list"><div><strong>01</strong><span>Paste any product review</span></div><div><strong>02</strong><span>Get a transparent confidence score</span></div><div><strong>03</strong><span>Turn feedback into your next decision</span></div></div>', unsafe_allow_html=True)
