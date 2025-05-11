# steps/data_loader.py

import pandas as pd
from zenml import step
from typing import Tuple
from sklearn.model_selection import train_test_split

@step
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed data from CSV files."""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test

