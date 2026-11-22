"""Test the fusion pipeline on the held-out test words.

    python models/fusion_pipeline/test.py

Writes Results/fusion_accuracy.md / .csv, Results/fusion_per_emotion_recall.csv,
Results/fusion_test_predictions.csv and Results/plots/{confusion,training,tsne}_<variant>.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.results import test_pipeline  # noqa: E402

VARIANTS = ["fusion_concat", "fusion_gated"]

if __name__ == "__main__":
    test_pipeline("fusion", VARIANTS)
