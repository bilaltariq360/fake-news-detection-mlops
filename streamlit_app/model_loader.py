# streamlit_app/model_loader.py

import joblib
import os

def load_model_and_vectorizer():
    model_path = os.path.join("../mlruns/models", "logistic_model.pkl")
    vectorizer_path = os.path.join("../mlruns/models", "tfidf_vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

