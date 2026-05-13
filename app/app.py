# ===============================
# Streamlit App - Sentiment Analysis
# ===============================

import os
import re
import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer

nltk_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "nltk_data")
os.environ["NLTK_DATA"] = nltk_data_path
nltk.data.path.append(nltk_data_path)

nltk.download("stopwords", download_dir=nltk_data_path)
nltk.download("wordnet", download_dir=nltk_data_path)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = ToktokTokenizer()

st.set_page_config(
    page_title="Sentiment Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "logreg_sentiment_model_full.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer_full.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model or vectorizer file not found! Make sure they exist in the 'models/' folder.")
        return None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model()

if model is None or vectorizer is None:
    st.stop()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

st.markdown(
    """
    <div class="hero">
        <div>
            <h1>Sentiment Explorer</h1>
            <p>Turn product reviews into clear sentiment insights with modern visuals.</p>
        </div>
        <div class="hero-badge">Interactive & reliable</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Quick tips")
st.sidebar.write("- Enter a complete review for best results")
st.sidebar.write("- Use the analysis button to see sentiment scores")
st.sidebar.write("- Try both positive and negative samples")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    review_input = st.text_area(
        "Enter review here:",
        height=220,
        placeholder="Type a product review and click Analyze Review"
    )
    if st.button("Analyze Review"):
        if not review_input.strip():
            st.warning("Please enter a review text!")
        else:
            clean_text = preprocess_text(review_input)
            vect_text = vectorizer.transform([clean_text])
            prediction = model.predict(vect_text)[0]
            prob = model.predict_proba(vect_text)[0]

            sentiment_label = "✅ Positive" if prediction == "positive" else "❌ Negative"
            color = "#10b981" if prediction == "positive" else "#ef4444"

            st.markdown(
                f"""
                <div class="result-card">
                  <h2>Prediction Results</h2>
                  <p class="result-status" style="color:{color};">{sentiment_label}</p>
                  <p><strong>Confidence</strong>: Positive {prob[1]:.2f}, Negative {prob[0]:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.bar_chart({"Positive": [prob[1]], "Negative": [prob[0]]})

with col2:
    st.image(
        "https://images.unsplash.com/photo-1517430816045-df4b7de2f4f6?auto=format&fit=crop&w=800&q=80",
        caption="Sentiment analysis for smarter product decisions",
        width=360,
    )
    st.markdown("### Why it works")
    st.write(
        "- Beautiful interface with cards and color accents\n"
        "- Quick results with confidence values\n"
        "- Designed for product review workflows"
    )
