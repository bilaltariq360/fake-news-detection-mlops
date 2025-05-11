# steps/evaluator.py

from zenml import step
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

@step
def evaluate_model(model: Pipeline, X_test, y_test) -> None:
    y_pred = model.predict(X_test["content"])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")

