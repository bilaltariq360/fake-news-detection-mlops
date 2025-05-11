# streamlit_app/app.py

import streamlit as st
from model_loader import load_model_and_vectorizer

# Load model
model, vectorizer = load_model_and_vectorizer()

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Enter a news title and content to detect if it's **Fake** or **Real**.")

# Input fields
title = st.text_input("News Title")
text = st.text_area("News Content")

if st.button("Predict"):
    if not title or not text:
        st.warning("Please enter both title and content.")
    else:
        combined_input = title + " " + text
        vectorized_input = vectorizer.transform([combined_input])
        prediction = model.predict(vectorized_input)[0]
        prob = model.predict_proba(vectorized_input)[0]

        label = "✅ Real" if prediction == 1 else "❌ Fake"
        confidence = max(prob) * 100

        st.subheader("Prediction:")
        st.markdown(f"**{label}**")
        st.text(f"Confidence: {confidence:.2f}%")

