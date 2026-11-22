"""Extra experiment: train on one speaker, test on the other.

With both speakers in training, the test set (new words, same two voices) is solved almost perfectly.
Here the model hears only one actress during training and is tested on the other actress's
recordings of the 30 test words: new words AND a new voice.

  direction   train / validation                      test
  OAF->YAF    OAF, train words / OAF, val words        YAF, test words (210 files)
  YAF->OAF    YAF, train words / YAF, val words        OAF, test words (210 files)

    python experiments/cross_speaker.py            # train (4 variants x 2 directions x 5 seeds) and test
    python experiments/cross_speaker.py --test     # test only, from saved checkpoints

Writes Results/cross_speaker_accuracy.md / .csv, Results/cross_speaker_test_predictions.csv,
Results/plots/confusion_cross_speaker_<variant>.png and tsne_cross_speaker_<variant>.png.
The text model is left out: its input is the same for every emotion, so it stays at chance.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import EMOTIONS, RESULTS  # noqa: E402
from src.pipeline import SEEDS, evaluate, manifest, train  # noqa: E402
from src.results import SHORT, plot_confusion, plot_tsne  # noqa: E402

VARIANTS = ["speech_wavlm", "speech_logmel", "fusion_concat", "fusion_gated"]
DIRECTIONS = [("OAF", "YAF"), ("YAF", "OAF")]


def split_for(df, src, dst):
    d = df.copy()
    d["split"] = "unused"
    d.loc[(d.speaker == src) & (df.split == "train"), "split"] = "train"
    d.loc[(d.speaker == src) & (df.split == "val"), "split"] = "val"
    d.loc[(d.speaker == dst) & (df.split == "test"), "split"] = "test"
    return d


def tag(src, dst):
    return "_xspk_%sto%s" % (src, dst)


def main():
    df = manifest()
    if "--test" not in sys.argv:
        for v in VARIANTS:
            for src, dst in DIRECTIONS:
                for s in SEEDS:
                    train(v, s, split_for(df, src, dst), tag(src, dst))
    rows, recalls, preds = [], [], []
    for v in VARIANTS:
        cm, accs = 0, {}
        for src, dst in DIRECTIONS:
            d = split_for(df, src, dst)
            runs = [evaluate(v, s, d, tag=tag(src, dst)) for s in SEEDS]
            accs[(src, dst)] = [r["acc"] for r in runs]
            cm = cm + sum(confusion_matrix(r["y"], r["pred"], labels=range(len(EMOTIONS))) for r in runs)
            for s, r in zip(SEEDS, runs):
                preds.append(pd.DataFrame({"variant": v, "direction": "%s->%s" % (src, dst), "seed": s,
                                           "file": r["file"], "true": [EMOTIONS[i] for i in r["y"]],
                                           "pred": [EMOTIONS[i] for i in r["pred"]],
                                           "confidence": r["prob"].max(1).round(3)}))
            if (src, dst) == DIRECTIONS[0]:
                plot_tsne(runs[0], v, df, name="cross_speaker_" + v, note="trained on %s, tested on %s" % (src, dst))
        all_acc = sum(accs.values(), [])
        rows.append({"variant": v, **{"%s->%s" % k: "%.1f%% ± %.1f" % (100 * np.mean(a), 100 * np.std(a))
                                     for k, a in accs.items()},
                     "both directions": "%.1f%%" % (100 * np.mean(all_acc))})
        recall = cm.diagonal() / cm.sum(1)
        recalls.append({"variant": v, **{SHORT[e]: "%.0f%%" % (100 * recall[i]) for i, e in enumerate(EMOTIONS)}})
        plot_confusion(cm, v, name="cross_speaker_" + v,
                       title="%s, cross-speaker: confusion (%%, 2 directions x 5 seeds)" % v)
    p = pd.concat(preds)
    p.to_csv(os.path.join(RESULTS, "cross_speaker_test_predictions.csv"), index=False)
    t, r = pd.DataFrame(rows), pd.DataFrame(recalls)
    t.to_csv(os.path.join(RESULTS, "cross_speaker_accuracy.csv"), index=False)
    f1 = p.groupby("variant").apply(lambda g: f1_score(g.true, g.pred, average="macro"), include_groups=False)
    md = ["# Cross-speaker test", "",
          "Trained on one speaker, tested on the other speaker's recordings of the 30 test words "
          "(210 files per direction). Mean ± std over 5 seeds. Chance: 14.3%.", "",
          "| Variant | OAF->YAF | YAF->OAF | Both directions | Macro F1 |", "|---|---|---|---|---|"]
    for row in rows:
        md.append("| %s | %s | %s | %s | %.3f |" % (row["variant"], row["OAF->YAF"], row["YAF->OAF"],
                                                   row["both directions"], f1[row["variant"]]))
    md += ["", "Recall per emotion (both directions, 5 seeds):", "",
           "| Variant | " + " | ".join(SHORT[e] for e in EMOTIONS) + " |", "|---" * (len(EMOTIONS) + 1) + "|"]
    md += ["| %s | " % x["variant"] + " | ".join(x[SHORT[e]] for e in EMOTIONS) + " |" for x in recalls]
    text = "\n".join(md) + "\n"
    with open(os.path.join(RESULTS, "cross_speaker_accuracy.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
