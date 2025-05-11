# pipelines/run_pipeline.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipelines.train_pipeline import training_pipeline

if __name__ == "__main__":
    training_pipeline()

