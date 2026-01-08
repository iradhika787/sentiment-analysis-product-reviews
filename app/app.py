# ===============================
# Streamlit App - Sentiment Analysis
# ===============================

import os
import pandas as pd
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer

# ===============================
# Step 0: Setup NLTK
# ===============================
import nltk
nltk_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "nltk_data")
os.environ["NLTK_DATA"] = nltk_data_path
nltk.data.path.append(nltk_data_path)

nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = ToktokTokenizer()

# ===============================
# Step 1: Load model & vectorizer
# ===============================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "logreg_sentiment_model_full.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer_full.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model or vectorizer file not found! Make sure they exist in 'models/' folder.")
        return None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model()

if model is None or vectorizer is None:
    st.stop()  # Stop execution if model not found

# ===============================
# Step 2: Preprocessing function
# ===============================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ===============================
# Step 3: Streamlit App UI
# ===============================
st.title("Sentiment Analysis - Product Reviews")
st.write("Enter a product review and get its predicted sentiment (Positive/Negative).")

review_input = st.text_area("Enter review here:")

if st.button("Predict Sentiment"):
    if not review_input.strip():
        st.warning("Please enter a review text!")
    else:
        clean_text = preprocess_text(review_input)
        vect_text = vectorizer.transform([clean_text])
        prediction = model.predict(vect_text)[0]
        prob = model.predict_proba(vect_text)[0]

        st.subheader("Prediction Results")
        st.write(f"**Sentiment:** {prediction.capitalize()}")
        st.write(f"**Probability:** Positive: {prob[1]:.2f}, Negative: {prob[0]:.2f}")
