"""Simple prosody measurements per speaker and emotion, used to explain the confusions.

    python experiments/prosody_stats.py

For every cleaned recording: duration, median pitch (F0) and pitch range (90th - 10th percentile)
over voiced frames. Loudness is not measured because preprocessing normalises every file to the
same RMS level. Writes Results/prosody_by_emotion.md and .csv
"""
import os
import sys

import librosa
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import DATA, EMOTIONS, RESULTS, SR  # noqa: E402


def measure(path):
    y, _ = librosa.load(path, sr=SR)
    f0, voiced, _ = librosa.pyin(y, fmin=75, fmax=600, sr=SR, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    if len(f0) == 0:
        return np.nan, np.nan
    return np.median(f0), np.percentile(f0, 90) - np.percentile(f0, 10)


def main():
    df = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    vals = [measure(os.path.join(DATA, "processed", f)) for f in df.file]
    df["f0_median"], df["f0_range"] = zip(*vals)
    df[["file", "speaker", "emotion", "sec", "f0_median", "f0_range"]].to_csv(
        os.path.join(RESULTS, "prosody_by_file.csv"), index=False)
    g = df.groupby(["emotion", "speaker"]).agg(sec=("sec", "mean"), f0=("f0_median", "mean"),
                                                 rng=("f0_range", "mean")).unstack()
    md = ["# Prosody per emotion and speaker", "",
          "Mean over all 200 words. Pitch from pYIN on voiced frames. Loudness is not shown: every file "
          "is normalised to the same RMS level in preprocessing.", "",
          "| Emotion | Duration OAF (s) | Duration YAF (s) | Median pitch OAF (Hz) | Median pitch YAF (Hz) "
          "| Pitch range OAF (Hz) | Pitch range YAF (Hz) |", "|---|---|---|---|---|---|---|"]
    for e in EMOTIONS:
        r = g.loc[e]
        md.append("| %s | %.2f | %.2f | %.0f | %.0f | %.0f | %.0f |" % (
            e, r[("sec", "OAF")], r[("sec", "YAF")], r[("f0", "OAF")], r[("f0", "YAF")],
            r[("rng", "OAF")], r[("rng", "YAF")]))
    text = "\n".join(md) + "\n"
    with open(os.path.join(RESULTS, "prosody_by_emotion.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
