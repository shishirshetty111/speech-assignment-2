"""Test the speech pipeline on the held-out test words.

    python models/speech_pipeline/test.py

Writes Results/speech_accuracy.md / .csv, Results/speech_per_emotion_recall.csv,
Results/speech_test_predictions.csv and Results/plots/{confusion,training,tsne}_<variant>.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.results import test_pipeline  # noqa: E402

VARIANTS = ["speech_wavlm", "speech_logmel"]

if __name__ == "__main__":
    test_pipeline("speech", VARIANTS)
