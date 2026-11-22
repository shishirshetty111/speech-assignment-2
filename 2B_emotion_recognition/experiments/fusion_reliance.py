"""How much do the fusion models rely on speech and on text?

    python experiments/fusion_reliance.py      (after the fusion and cross-speaker models are trained)

For each fusion model, on its test set:
  normal          the real speech and the real text of each file
  text swapped    each file gets the text of another, randomly chosen test file
  speech swapped  each file gets the speech of another, randomly chosen test file
  gate            gated fusion only: mean value of the gate, i.e. the share of the fused vector that
                  comes from speech (1 = speech only, 0 = text only)
If accuracy does not change when the text is swapped, the model ignores the text.
Writes Results/fusion_reliance.md
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import RESULTS  # noqa: E402
from src.pipeline import SEEDS, Batches, features, load, manifest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_speaker import DIRECTIONS, split_for, tag  # noqa: E402


def swapped(name, files, seed):
    feats = features(name)
    other = list(files)
    random.Random(seed).shuffle(other)
    return {f: feats[o] for f, o in zip(files, other)}


@torch.no_grad()
def accuracy(model, ck, test, **override):
    model.eval()
    correct, n, gates = 0, 0, []
    for b in Batches(test, ck["cfg"], ck["norm"], **override):
        correct += (model(b).argmax(-1) == b["y"]).sum().item()
        n += len(b["y"])
        if hasattr(model.fusion, "last_gate"):
            gates.append(model.fusion.last_gate.mean().item())
    return correct / n, (np.mean(gates) if gates else float("nan"))


def main():
    df = manifest()
    settings = [("both speakers", "", df)] + [("cross-speaker %s->%s" % (s, d), tag(s, d), split_for(df, s, d))
                                              for s, d in DIRECTIONS]
    md = ["# Reliance of the fusion models on each modality", "",
          "Mean over 5 seeds. Swapped: each test file is given the text (or speech) of another test file.", "",
          "| Model | Test setting | Normal | Text swapped | Speech swapped | Gate (share from speech) |",
          "|---|---|---|---|---|---|"]
    for variant in ("fusion_concat", "fusion_gated"):
        for label, t, d in settings:
            test = d[d.split == "test"]
            res = []
            for s in SEEDS:
                model, ck = load(variant, s, t)
                acc, gate = accuracy(model, ck, test)
                acc_t, _ = accuracy(model, ck, test, text_override=swapped("text", test.file, s))
                acc_s, _ = accuracy(model, ck, test, speech_override=swapped("wavlm", test.file, s))
                res.append((acc, acc_t, acc_s, gate))
            m = np.mean(res, axis=0)
            md.append("| %s | %s | %.1f%% | %.1f%% | %.1f%% | %s |" % (
                variant, label, 100 * m[0], 100 * m[1], 100 * m[2], "-" if np.isnan(m[3]) else "%.2f" % m[3]))
    text = "\n".join(md) + "\n"
    with open(os.path.join(RESULTS, "fusion_reliance.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
