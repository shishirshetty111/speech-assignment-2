"""Train the speech pipeline (speech only: WavLM or log-mel features, BiLSTM + attention, classifier).

    python models/speech_pipeline/train.py                # all variants, seeds 0-4
    python models/speech_pipeline/train.py --seeds 0      # one seed only (quick check)

Checkpoints: checkpoints/<variant>_seed<n>.pt (best validation accuracy).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.pipeline import SEEDS, train  # noqa: E402

VARIANTS = ["speech_wavlm", "speech_logmel"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args()
    for v in args.variants:
        for s in args.seeds:
            train(v, s)
