"""Step 4: align the corpus with each dictionary / acoustic model combination.

Each experiment writes one TextGrid per file to outputs/textgrids/<experiment>/<speaker>/.

  experiment     dictionary                         acoustic model
  arpa_baseline  english_us_arpa, OOVs left as-is   english_us_arpa (pretrained)
  arpa_g2p       english_us_arpa + pretrained G2P   english_us_arpa (pretrained)      <- main result
  arpa_own_g2p   english_us_arpa + our own G2P      english_us_arpa (pretrained)
  mfa_g2p        english_mfa + pretrained G2P       english_mfa (pretrained, IPA-style phones)
  arpa_learner   as arpa_g2p + learner variants      english_us_arpa (pretrained)  (learner_variants.py)
  arpa_adapted   english_us_arpa + pretrained G2P   english_us_arpa adapted to this corpus (mfa adapt)
  scratch        english_us_arpa + pretrained G2P   trained from scratch on this corpus only (mfa train)

    python scripts/align_all.py                 # all experiments
    python scripts/align_all.py --only arpa_g2p
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CORPUS, TEXTGRIDS, WORK, run_mfa  # noqa: E402

D = os.path.join(WORK, "dicts")
MODELS = os.path.join(WORK, "models")

# name -> (mode, dictionary, acoustic model or None)
EXPERIMENTS = {
    "arpa_baseline": ("align", "english_us_arpa", "english_us_arpa"),
    "arpa_g2p": ("align", os.path.join(D, "arpa_pretrained_g2p.dict"), "english_us_arpa"),
    "arpa_own_g2p": ("align", os.path.join(D, "arpa_own_g2p.dict"), "english_us_arpa"),
    "mfa_g2p": ("align", os.path.join(D, "mfa_pretrained_g2p.dict"), "english_mfa"),
    "arpa_learner": ("align", os.path.join(D, "arpa_learner_variants.dict"), "english_us_arpa"),
    "arpa_adapted": ("adapt", os.path.join(D, "arpa_pretrained_g2p.dict"), "english_us_arpa"),
    "scratch": ("train", os.path.join(D, "arpa_pretrained_g2p.dict"), None),
}


def run(name, jobs):
    mode, dictionary, acoustic = EXPERIMENTS[name]
    out = os.path.join(TEXTGRIDS, name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    tmp = os.path.join(WORK, "align_" + name)
    log = os.path.join(WORK, "logs", "align_%s.log" % name)
    common = ["--clean", "-j", str(jobs), "--temporary_directory", tmp]
    if mode == "align":
        run_mfa("align", CORPUS, dictionary, acoustic, out, "--overwrite", *common, log=log)
    elif mode == "adapt":
        # `mfa adapt --output_directory` exports the alignment made BEFORE adaptation, so the
        # adapted model is saved first and then used in a normal `mfa align` run.
        adapted = os.path.join(MODELS, "acoustic_arpa_adapted.zip")
        run_mfa("adapt", CORPUS, dictionary, acoustic, adapted, "--overwrite", *common, log=log)
        run_mfa("align", CORPUS, dictionary, adapted, out, "--overwrite", "--clean", "-j", str(jobs),
                "--temporary_directory", tmp + "_align", log=log.replace(".log", "_align.log"))
    else:
        run_mfa("train", CORPUS, dictionary, os.path.join(MODELS, "acoustic_scratch.zip"),
                "--output_directory", out, *common, log=log)
    n = sum(f.endswith(".TextGrid") for _, _, fs in os.walk(out) for f in fs)
    print("%s: %d TextGrids in %s" % (name, n, out))
    if n == 0:
        sys.exit("no TextGrids written for " + name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(EXPERIMENTS))
    ap.add_argument("-j", "--jobs", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(MODELS, exist_ok=True)
    for name in args.only or EXPERIMENTS:
        run(name, args.jobs)


if __name__ == "__main__":
    main()
