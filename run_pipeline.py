"""Run the whole forced-alignment pipeline with one command.

    python run_pipeline.py                 # everything
    python run_pipeline.py --skip_g2p_training   # reuse work/models/g2p_own_arpa.zip if present,
                                                  # otherwise leave out the arpa_own_g2p experiment

Steps
  1. prepare_corpus      data/raw -> data/corpus (MFA layout, SPHERE -> WAV, transcript clean-up)
  2. download models     MFA pretrained acoustic models, dictionaries and G2P models
  3. train_g2p           our own G2P model (extra credit; about 30 min on CPU)
     evaluate_g2p        own vs pretrained G2P on 2,000 unseen dictionary words
  4. build_dictionaries  base dictionary + pronunciations for missing words
  5. align_all           one TextGrid set per dictionary / acoustic model combination
     learner_variants    dictionary with learner pronunciations for the ISLE minimal pairs
  6. analyse             statistics and boundary comparison (outputs/analysis)
  7. make_figures        Praat pictures (outputs/figures)

Run it with the Python of the MFA conda environment (see README).
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from common import WORK, run_mfa  # noqa: E402

MODELS = [("acoustic", "english_us_arpa"), ("acoustic", "english_mfa"), ("dictionary", "english_us_arpa"),
          ("dictionary", "english_mfa"), ("g2p", "english_us_arpa"), ("g2p", "english_us_mfa")]


def step(script, *args):
    print("\n=== %s %s" % (script, " ".join(args)), flush=True)
    res = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), *args])
    if res.returncode != 0:
        sys.exit("step %s failed" % script)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_g2p_training", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=2)
    args = ap.parse_args()

    step("prepare_corpus.py")
    print("\n=== downloading MFA models", flush=True)
    for kind, name in MODELS:
        run_mfa("model", "download", kind, name, "--ignore_cache")
    own_g2p = os.path.join(WORK, "models", "g2p_own_arpa.zip")
    if not args.skip_g2p_training:
        step("train_g2p.py", "-j", str(args.jobs))
    if os.path.exists(own_g2p):
        step("evaluate_g2p.py")
    variants = ["arpa_pretrained_g2p", "mfa_pretrained_g2p"] + (["arpa_own_g2p"] if os.path.exists(own_g2p) else [])
    for v in variants:
        step("build_dictionaries.py", "--variant", v)
    experiments = ["arpa_baseline", "arpa_g2p", "mfa_g2p", "arpa_adapted", "scratch"]
    if os.path.exists(own_g2p):
        experiments.insert(2, "arpa_own_g2p")
    step("align_all.py", "-j", str(args.jobs), "--only", *experiments)
    step("learner_variants.py")
    step("analyse.py")
    step("make_figures.py")
    print("\nDone. TextGrids: outputs/textgrids/  analysis: outputs/analysis/  figures: outputs/figures/")


if __name__ == "__main__":
    main()
