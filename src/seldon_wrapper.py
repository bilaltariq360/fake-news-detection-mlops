import joblib
import numpy as np

class FakeNewsClassifier:
    def __init__(self):
        self.model = joblib.load("../mlruns/models/logistic_model.pkl")
        self.vectorizer = joblib.load("../mlruns/models/tfidf_vectorizer.pkl")

    def predict(self, X, features_names):
        X_transformed = self.vectorizer.transform(X)
        preds = self.model.predict(X_transformed)
        return preds.tolist()

