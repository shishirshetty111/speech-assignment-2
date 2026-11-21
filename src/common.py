"""Paths and constants shared by all scripts."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FEATURES = os.path.join(ROOT, "features")
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "Results")
PLOTS = os.path.join(RESULTS, "plots")

SR = 16000
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
