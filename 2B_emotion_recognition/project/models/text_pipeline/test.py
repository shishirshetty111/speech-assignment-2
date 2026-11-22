"""Test the text pipeline on the held-out test words.

    python models/text_pipeline/test.py

Writes Results/text_accuracy.md / .csv, Results/text_per_emotion_recall.csv,
Results/text_test_predictions.csv and Results/plots/{confusion,training,tsne}_<variant>.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.results import test_pipeline  # noqa: E402

VARIANTS = ["text_bert"]

if __name__ == "__main__":
    test_pipeline("text", VARIANTS)
