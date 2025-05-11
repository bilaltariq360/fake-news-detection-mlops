# src/train.py

import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import joblib

def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    return X_train.squeeze(), X_test.squeeze(), y_train.squeeze(), y_test.squeeze()

def train_and_log():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Define and train model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save model
    joblib.dump(model, os.path.join("./mlruns/models", "logistic_model.pkl"))

    # Save vectorizer
    joblib.dump(vectorizer, os.path.join("./mlruns/models", "tfidf_vectorizer.pkl"))

    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")

    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_features", 5000)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save and log the vectorizer as an artifact
        joblib.dump(vectorizer, "vectorizer.joblib")
        mlflow.log_artifact("vectorizer.joblib")

        print("✅ Model, vectorizer, and metrics logged to MLflow.")

if __name__ == "__main__":
    train_and_log()

