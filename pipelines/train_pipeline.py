# pipelines/train_pipeline.py

from zenml import pipeline
from steps.data_loader import load_data
from steps.trainer import train_model
from steps.evaluator import evaluate_model

@pipeline
def training_pipeline():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train=X_train, y_train=y_train)
    evaluate_model(model=model, X_test=X_test, y_test=y_test)

