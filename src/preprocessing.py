import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess(input_path, output_dir):
    df = pd.read_csv(input_path)
    df = df[['title', 'text', 'label']].dropna()
    df['content'] = df['title'] + " " + df['text']

    X_train, X_test, y_train, y_test = train_test_split(
        df['content'], df['label'], test_size=0.2, random_state=42
    )

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("✅ Data preprocessed and saved successfully!")

if __name__ == "__main__":
    preprocess("../data/raw/fake_news.csv", "../data/processed")

