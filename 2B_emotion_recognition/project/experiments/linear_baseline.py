"""Simple baseline: time-averaged WavLM features -> logistic regression.

    python experiments/linear_baseline.py     (needs features/wavlm_layer_means.npy from extract_features.py)

Same test sets as the main and cross-speaker experiments:
  both speakers   train words of both speakers -> test words of both speakers
  OAF->YAF        OAF train words -> YAF test words      (and YAF->OAF)
The regularisation strength C is chosen on the validation words of the training speaker(s) only;
no label of the test speaker is used. The model is deterministic, so there are no seeds.

Main result: layer 9 (the layer the neural models use). The second table repeats the cross-speaker
test for every layer. It is a sensitivity check only and is not used to choose anything: layer 9 was
chosen earlier with a probe that validated on the other speaker's validation words, so the
cross-speaker results for layer 9 are slightly optimistic, and this table shows by how much the
layer choice matters.
Writes Results/linear_baseline.md
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import FEATURES, RESULTS  # noqa: E402
from src.pipeline import manifest  # noqa: E402

LAYER = 9
CS = [0.001, 0.01, 0.1, 1.0, 10.0]


def fit_predict(X, y, tr, va, te):
    """Returns test accuracy with the C chosen on validation (ties -> smallest C), that C, and the
    test accuracy of every C (to show how much the choice of C matters)."""
    best, tests = None, []
    sc = StandardScaler().fit(X[tr])
    for c in CS:
        clf = LogisticRegression(C=c, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        acc = clf.score(sc.transform(X[va]), y[va])
        tests.append(clf.score(sc.transform(X[te]), y[te]))
        if best is None or acc > best[0]:
            best = (acc, c, tests[-1])
    return best[2], best[1], tests


def settings(df):
    s = df.split.values
    yield "both speakers", s == "train", s == "val", s == "test"
    for a, b in (("OAF", "YAF"), ("YAF", "OAF")):
        src, dst = (df.speaker == a).values, (df.speaker == b).values
        yield "%s->%s" % (a, b), src & (s == "train"), src & (s == "val"), dst & (s == "test")


def main():
    df = manifest()
    pooled = np.load(os.path.join(FEATURES, "wavlm_layer_means.npy"))
    y = df.emotion.values
    md = ["# Linear baseline: time-averaged WavLM -> logistic regression", "",
          "C chosen on the validation words of the training speaker(s) only. Deterministic (no seeds).", "",
          "On the training speaker's validation words every C reaches about 100%, so the tie goes to",
          "the smallest C. The last column shows the test accuracy for every C in %s." % CS, "",
          "| Test setting | Layer 9 test accuracy | C | Range over all C |", "|---|---|---|---|"]
    for name, tr, va, te in settings(df):
        acc, c, tests = fit_predict(pooled[:, LAYER], y, tr, va, te)
        md.append("| %s | %.1f%% | %g | %.1f-%.1f%% |" % (name, 100 * acc, c, 100 * min(tests), 100 * max(tests)))
        print(name, "%.3f" % acc, c, flush=True)
    md += ["", "Sensitivity check, cross-speaker test accuracy per layer (not used to choose anything):", "",
           "| Layer | OAF->YAF | YAF->OAF | Mean |", "|---|---|---|---|"]
    cross = [x for x in settings(df)][1:]
    for layer in range(pooled.shape[1]):
        accs = [fit_predict(pooled[:, layer], y, tr, va, te)[0] for _, tr, va, te in cross]
        md.append("| %d | %.1f%% | %.1f%% | %.1f%% |" % (layer, 100 * accs[0], 100 * accs[1], 100 * np.mean(accs)))
        print("layer", layer, ["%.3f" % a for a in accs], flush=True)
    text = "\n".join(md) + "\n"
    with open(os.path.join(RESULTS, "linear_baseline.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
