# steps/trainer.py

from zenml import step
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

@step
def train_model(X_train, y_train) -> Pipeline:
    """Train a simple Logistic Regression model."""
    vectorizer = TfidfVectorizer()
    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", model)
    ])

    pipeline.fit(X_train["content"], y_train)
    return pipeline

