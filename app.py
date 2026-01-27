import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("🕵️ Fake Job Detection System")

job_desc = st.text_area("Paste Job Description Here")

if st.button("Check Job"):
    if job_desc.strip() == "":
        st.warning("Please enter a job description")
    else:
        cleaned = clean_text(job_desc)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.error("❌ Fake Job Posting")
        else:
            st.success("✅ Legitimate Job Posting")