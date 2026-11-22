"""Test-set evaluation, tables and plots for one pipeline (called by models/*/test.py)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, silhouette_score  # noqa: E402

from src.common import EMOTIONS, PLOTS, RESULTS  # noqa: E402
from src.pipeline import SEEDS, VARIANTS, evaluate, load, manifest  # noqa: E402

SHORT = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy", "neutral": "neutral",
         "pleasant_surprise": "surprise", "sad": "sad"}
BLOCK = {"speech": "temporal modelling", "text": "contextual modelling", "fusion": "fusion"}
COLORS = ["#d62728", "#8c564b", "#9467bd", "#ff7f0e", "#7f7f7f", "#17becf", "#1f77b4"]


def test_pipeline(name, variants):
    os.makedirs(PLOTS, exist_ok=True)
    df = manifest()
    rows, per_emotion, preds = [], [], None
    for v in variants:
        runs = [evaluate(v, s, df) for s in SEEDS]
        accs = [r["acc"] for r in runs]
        f1s = [f1_score(r["y"], r["pred"], average="macro") for r in runs]
        vals = [load(v, s)[1]["best_val_acc"] for s in SEEDS]
        rows.append({"variant": v, "test accuracy (mean)": np.mean(accs), "test accuracy (std)": np.std(accs),
                     "macro F1 (mean)": np.mean(f1s), "validation accuracy (mean)": np.mean(vals),
                     "test accuracy per seed": " ".join("%.3f" % a for a in accs)})
        cm = sum(confusion_matrix(r["y"], r["pred"], labels=range(len(EMOTIONS))) for r in runs)
        recall = cm.diagonal() / cm.sum(1)
        per_emotion.append({"variant": v, **{SHORT[e]: recall[i] for i, e in enumerate(EMOTIONS)}})
        plot_confusion(cm, v)
        plot_history(v)
        rows[-1]["silhouette (block output, seed 0)"] = plot_tsne(runs[0], v, df)
        p = pd.DataFrame({"file": runs[0]["file"], "true": [EMOTIONS[i] for i in runs[0]["y"]]})
        for s, r in zip(SEEDS, runs):
            p["%s_seed%d" % (v, s)] = [EMOTIONS[i] for i in r["pred"]]
        p["%s_confidence_seed0" % v] = runs[0]["prob"].max(1).round(3)
        preds = p if preds is None else preds.merge(p.drop(columns="true"), on="file")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "%s_accuracy.csv" % name), index=False)
    pd.DataFrame(per_emotion).to_csv(os.path.join(RESULTS, "%s_per_emotion_recall.csv" % name), index=False)
    preds.to_csv(os.path.join(RESULTS, "%s_test_predictions.csv" % name), index=False)
    md = ["| Variant | Test accuracy (mean ± std, 5 seeds) | Macro F1 | Validation accuracy | Silhouette |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %.1f%% ± %.1f | %.3f | %.1f%% | %.2f |" % (
            r["variant"], 100 * r["test accuracy (mean)"], 100 * r["test accuracy (std)"], r["macro F1 (mean)"],
            100 * r["validation accuracy (mean)"], r["silhouette (block output, seed 0)"]))
    md += ["", "Recall per emotion (share of each emotion's test files classified correctly, all 5 seeds):", "",
           "| Variant | " + " | ".join(SHORT[e] for e in EMOTIONS) + " |", "|---" * (len(EMOTIONS) + 1) + "|"]
    for r in per_emotion:
        md.append("| %s | " % r["variant"] + " | ".join("%.0f%%" % (100 * r[SHORT[e]]) for e in EMOTIONS) + " |")
    text = "\n".join(md) + "\n"
    with open(os.path.join(RESULTS, "%s_accuracy.md" % name), "w", encoding="utf-8", newline="\n") as f:
        f.write("# %s pipeline: test results\n\nTest set: 30 words never seen in training, both speakers, "
                "all 7 emotions (420 files). Chance level: 14.3%%. Silhouette: how well the emotions form "
                "separate clusters in the block output (-1 to 1, higher is better).\n\n" % name.capitalize() + text)
    print(text)


def plot_confusion(cm, variant, name=None, title=None):
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = [SHORT[e] for e in EMOTIONS]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if norm[i, j] >= 0.005:
                ax.text(j, i, "%.0f" % (100 * norm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or "%s: test confusion (%%, 5 seeds)" % variant, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "confusion_%s.png" % (name or variant)), dpi=150)
    plt.close(fig)


def plot_history(variant):
    h = pd.DataFrame(load(variant, 0)[1]["history"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.2))
    a.plot(h.epoch, h.train_loss, label="train")
    a.plot(h.epoch, h.val_loss, label="validation")
    a.set_title("loss")
    b.plot(h.epoch, h.train_acc, label="train")
    b.plot(h.epoch, h.val_acc, label="validation")
    b.set_title("accuracy")
    for ax in (a, b):
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle("%s, seed 0" % variant, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "training_%s.png" % variant), dpi=150)
    plt.close(fig)


def plot_tsne(run, variant, df, name=None, note="test set"):
    """2-D t-SNE map of the block representation for the test files (seed 0 model)."""
    speakers = df.set_index("file").loc[run["file"], "speaker"].values
    rep = run["rep"]
    sil = silhouette_score(rep, run["y"]) if len(set(run["y"])) > 1 else float("nan")
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(rep)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for i, e in enumerate(EMOTIONS):
        for spk, marker in (("OAF", "o"), ("YAF", "^")):
            m = (run["y"] == i) & (speakers == spk)
            ax.scatter(xy[m, 0], xy[m, 1], s=12, c=COLORS[i], marker=marker, alpha=0.8,
                       label=SHORT[e] if spk == "OAF" else None, edgecolors="none")
    ax.legend(fontsize=7, markerscale=1.5, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    kind = VARIANTS[variant]["kind"]
    ax.set_title("%s: %s output (%s)\ncircle = older speaker, triangle = younger; silhouette %.2f"
                 % (variant, BLOCK[kind], note, sil), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "tsne_%s.png" % (name or variant)), dpi=150)
    plt.close(fig)
    return sil
