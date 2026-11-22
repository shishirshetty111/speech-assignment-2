"""Extra credit: train our own G2P (grapheme-to-phoneme) model with MFA.

The model learns spelling -> ARPAbet pronunciation from the english_us_arpa dictionary
(about 200,000 entries; by default a fixed random 50,000-word sample). With --evaluate, MFA holds
out 10% of the sample and trains on the other 90%. Its own score on the held-out part came out as
100% WER / 100% PER although the model predicts correct pronunciations, so the model is scored
separately by evaluate_g2p.py. The model is used by build_dictionaries.py (variant arpa_own_g2p)
to pronounce the corpus words missing from the dictionary.

    python scripts/train_g2p.py                # 50,000 random words, 2 jobs
    python scripts/train_g2p.py --words 0      # whole dictionary (needs much more memory)
"""
import argparse
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dictionaries import base_dict_path  # noqa: E402
from common import ANALYSIS, WORK, run_mfa  # noqa: E402

MODEL = os.path.join(WORK, "models", "g2p_own_arpa.zip")


def subset_dictionary(n_words, seed=0):
    """Write a dictionary with n_words randomly chosen words (all their pronunciations kept).
    The selection is fixed by the seed, so reruns train on the same words."""
    with open(base_dict_path("english_us_arpa"), encoding="utf-8") as f:
        lines = [l for l in f if "\t" in l]
    words = sorted({l.split("\t", 1)[0] for l in lines})
    keep = set(random.Random(seed).sample(words, min(n_words, len(words))))
    path = os.path.join(WORK, "dicts", "g2p_train_%d.dict" % len(keep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(l for l in lines if l.split("\t", 1)[0] in keep)
    return path, len(keep), len(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--jobs", type=int, default=2)
    ap.add_argument("--random_starts", type=int, default=5,
                    help="letter-phone alignment restarts (MFA default 25); the best one is kept")
    ap.add_argument("--words", type=int, default=50000,
                    help="train on this many randomly chosen dictionary words (0 = all); "
                         "the full dictionary needs much more memory and time")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    if args.words:
        train_dict, n, total = subset_dictionary(args.words)
    else:
        train_dict = base_dict_path("english_us_arpa")
        n = total = None
    config = os.path.join(WORK, "g2p_config.yaml")
    with open(config, "w", encoding="utf-8", newline="\n") as f:
        f.write("random_starts: %d\n" % args.random_starts)
    t = time.time()
    out = run_mfa("train_g2p", train_dict, MODEL, "--clean", "--evaluate", "-c", config,
                  "--validation_proportion", "0.1", "-j", str(args.jobs),
                  "--temporary_directory", os.path.join(WORK, "g2p_tmp"),
                  log=os.path.join(WORK, "logs", "train_g2p.log"))
    wer = re.findall(r"WER:?\s*([\d.]+)", out)
    per = re.findall(r"(?:PER|LER):?\s*([\d.]+)", out)
    os.makedirs(ANALYSIS, exist_ok=True)
    with open(os.path.join(ANALYSIS, "g2p_training.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("training data: %s words of the english_us_arpa dictionary, 10%% of them held out\n"
                "MFA built-in held-out WER: %s (not reliable, see evaluate_g2p.py)\nMFA built-in held-out PER: %s\n"
                "training time: %.1f min (%d jobs, %d random starts)\n"
                % ("%d of %d" % (n, total) if n else "all", wer[-1] if wer else "see work/logs/train_g2p.log",
                   per[-1] if per else "see log", (time.time() - t) / 60, args.jobs, args.random_starts))
    print(open(os.path.join(ANALYSIS, "g2p_training.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
