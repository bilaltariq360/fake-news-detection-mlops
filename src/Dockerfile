FROM python:3.9-slim

WORKDIR /app

COPY fake-news-mlops/mlruns/models /models
COPY fake-news-mlops/src/seldon_wrapper.py .

RUN pip install joblib scikit-learn numpy seldon-core

EXPOSE 5000

ENV MODEL_NAME=FakeNewsClassifier

CMD ["seldon-core-microservice", "seldon_wrapper", "FakeNewsClassifier"]

