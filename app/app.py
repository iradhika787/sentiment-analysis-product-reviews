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
    <div class="topbar">
        <div class="brand-lockup"><span class="brand-mark">RS</span><span>ReviewSignal</span></div>
        <div class="live-status"><span class="status-dot"></span> ANALYSIS ENGINE ONLINE</div>
    </div>
    <div class="hero">
        <div class="hero-content">
            <div class="hero-kicker">CUSTOMER INTELLIGENCE / 01</div>
            <h1>Find the feeling<br><em>behind the feedback.</em></h1>
            <p>Turn unstructured product reviews into a clear signal your team can act on.</p>
        </div>
        <div class="hero-stamp"><span>LIVE</span><strong>01</strong><small>Review<br>analyzer</small></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="trust-strip">
        <div><span class="strip-number">01</span><div><strong>Instant signal</strong><small>Results in under a second</small></div></div>
        <div><span class="strip-number">02</span><div><strong>Clear confidence</strong><small>See the strength of every read</small></div></div>
        <div><span class="strip-number">03</span><div><strong>Private by design</strong><small>Your review is not stored</small></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-brand"><span class="brand-mark">RS</span><div><strong>ReviewSignal</strong><small>Product intelligence</small></div></div>', unsafe_allow_html=True)
    st.markdown("#### Start with a sample")
    samples = {
        "Positive review": "The fabric feels premium and the fit is exactly as described. I would buy this again.",
        "Critical review": "The color looked different from the photos and the stitching started to unravel after one wash.",
    }
    selected_sample = st.selectbox("Load an example", ["Choose a sample"] + list(samples), label_visibility="collapsed")
    st.divider()
    st.markdown("#### A calm read on noisy feedback")
    st.caption("Your review is cleaned, converted into TF-IDF features, and scored by a trained logistic regression model.")
    st.caption("No review text is stored by this app.")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown('<div class="section-heading"><div><span class="section-index">01</span><h2>Review workspace</h2></div><span class="section-note">Paste, scan, decide</span></div>', unsafe_allow_html=True)
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
    analyze = st.button("Run sentiment analysis  →", type="primary", use_container_width=True)
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
            confidence_note = "A strong directional signal." if confidence >= 0.75 else "A mixed signal; pair with human review."

            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-topline"><div class="result-label">SENTIMENT RESULT</div><span class="result-chip" style="color:{color}; background:{color}18;">{sentiment_label.upper()}</span></div>
                  <div class="result-main"><div><h2 style="color:{color};">{sentiment_label}</h2><p class="result-status">{confidence_note}</p></div><div class="confidence-ring" style="--score:{confidence * 360}deg; --ring-color:{color};"><strong>{confidence:.0%}</strong><small>confidence</small></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            metric1, metric2 = st.columns(2)
            metric1.metric("Positive signal", f"{positive_score:.0%}")
            metric2.metric("Negative signal", f"{negative_score:.0%}")
            st.progress(confidence, text=f"Model confidence: {confidence:.0%}")

with col2:
    st.markdown('<div class="insight-panel"><div class="panel-eyebrow">PRODUCT TEAM VIEW</div><h2>From words<br>to direction.</h2><p>Use sentiment as an early signal for what customers love, what frustrates them, and where the product experience needs attention.</p><div class="insight-rule"></div><small>Best used alongside review volume, product area, and customer context.</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading compact"><div><span class="section-index">02</span><h2>Simple by design</h2></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-list"><div><strong>01</strong><span>Paste any product review</span></div><div><strong>02</strong><span>Get a transparent confidence score</span></div><div><strong>03</strong><span>Turn feedback into your next decision</span></div></div>', unsafe_allow_html=True)
