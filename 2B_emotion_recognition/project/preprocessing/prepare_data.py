"""Preprocessing block (both modalities) and the train / validation / test split.

    python preprocessing/prepare_data.py

Input : data/raw/**/<speaker>_<word>_<emotion>.wav   (TESS from Kaggle; any folder layout)
Output: data/processed/<speaker>_<word>_<emotion>.wav  cleaned speech (16 kHz mono)
        data/manifest.csv                              one row per utterance: labels, text, split
        Results/data_summary.md                        counts and audio statistics

Speech preprocessing
  - read, mix to mono, resample to 16 kHz (the rate WavLM was trained on);
  - trim leading and trailing silence (librosa.effects.trim, 30 dB below the peak), keeping
    0.05 s on each side so the first and last sounds are not cut;
  - normalise loudness to the same RMS level, so the model cannot use recording level as a cue;
  - lengths are left variable here; the models handle them with padding and masks.
Text preprocessing
  - TESS has no transcript files: every recording is the carrier phrase "Say the word <word>",
    so the transcript is built from the file name, lower-cased and stripped of punctuation.
    Tokenisation is done by the BERT tokenizer in extract_features.py.

Split: by WORD, not by file. The 200 target words are shuffled with a fixed seed and divided
140 / 30 / 30 into train / validation / test. Every word is spoken by both speakers in all 7
emotions, so a random file split would put the same word, from the same speaker, in both training
and test. The word split tests on words the models have never heard.
"""
import hashlib
import os
import random
import sys
from collections import Counter

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.common import DATA, EMOTIONS, RESULTS, SR  # noqa: E402

RAW = os.path.join(DATA, "raw")
PROCESSED = os.path.join(DATA, "processed")
LABEL_FIX = {"ps": "pleasant_surprise", "pleasant_surprised": "pleasant_surprise", "fear": "fear"}
TARGET_RMS = 0.05
SEED = 0


def find_files():
    """All wav files, keeping one copy when the archive holds the same file twice."""
    by_name, dup_same, dup_diff = {}, 0, 0
    for d, _, files in os.walk(RAW):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            path = os.path.join(d, f)
            name = f[:-4].lower()
            if name in by_name:
                same = _md5(path) == _md5(by_name[name])
                dup_same += same
                dup_diff += not same
                continue
            by_name[name] = path
    return by_name, dup_same, dup_diff


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def clean(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    raw_dur = len(y) / SR
    _, (a, b) = librosa.effects.trim(y, top_db=30)
    pad = int(0.05 * SR)
    y = y[max(0, a - pad): min(len(y), b + pad)]
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    y = y * (TARGET_RMS / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:  # avoid clipping after the gain
        y = y * (0.99 / peak)
    return y, sr, raw_dur


def main():
    files, dup_same, dup_diff = find_files()
    if dup_diff:
        sys.exit("%d files share a name but differ in content; check data/raw" % dup_diff)
    os.makedirs(PROCESSED, exist_ok=True)
    rows, notes = [], []
    for name, path in sorted(files.items()):
        speaker, word, emotion = name.split("_", 2)
        folder_speaker = os.path.basename(os.path.dirname(path)).split("_")[0].upper()
        if speaker.upper() != folder_speaker:  # e.g. OA_bite_neutral.wav inside OAF_neutral/
            notes.append("%s: speaker taken from the folder name (%s)" % (os.path.basename(path), folder_speaker))
            speaker = folder_speaker
        emotion = LABEL_FIX.get(emotion, emotion)
        if emotion not in EMOTIONS:
            sys.exit("unknown emotion label in %s" % path)
        y, sr, raw_dur = clean(path)
        if sr != 24414:
            notes.append("%s: recorded at %d Hz (others 24414 Hz), resampled like the rest"
                         % (os.path.basename(path), sr))
        out = os.path.join(PROCESSED, "%s_%s_%s.wav" % (speaker.upper(), word, emotion))
        sf.write(out, y, SR, subtype="PCM_16")
        rows.append(dict(file=os.path.basename(out), speaker=speaker.upper(), word=word, emotion=emotion,
                         text="say the word %s" % word, orig_sr=sr, raw_sec=round(raw_dur, 3),
                         sec=round(len(y) / SR, 3)))
    df = pd.DataFrame(rows)

    words = sorted(df.word.unique())
    random.Random(SEED).shuffle(words)
    split = {w: "train" for w in words[:140]}
    split.update({w: "val" for w in words[140:170]})
    split.update({w: "test" for w in words[170:]})
    df["split"] = df.word.map(split)
    df.to_csv(os.path.join(DATA, "manifest.csv"), index=False)

    # checks and summary
    combos = df.groupby("word").apply(lambda g: len(set(zip(g.speaker, g.emotion))), include_groups=False)
    lines = ["# Data summary", "",
             "Files found: %d unique recordings (%d exact duplicate copies in the archive were skipped)."
             % (len(df), dup_same), "",
             "| | Count |", "|---|---|",
             "| Speakers | %s |" % ", ".join("%s (%d)" % kv for kv in sorted(Counter(df.speaker).items())),
             "| Emotions | %d |" % df.emotion.nunique(),
             "| Words | %d |" % len(words),
             "| Words recorded by both speakers in all 7 emotions | %d |" % int((combos == 14).sum()),
             "| Original sample rates | %s |" % ", ".join("%d Hz (%d)" % kv for kv in sorted(Counter(df.orig_sr).items())),
             "| Length before trimming (s) | min %.2f, median %.2f, max %.2f |" % (df.raw_sec.min(), df.raw_sec.median(), df.raw_sec.max()),
             "| Length after trimming (s) | min %.2f, median %.2f, max %.2f |" % (df.sec.min(), df.sec.median(), df.sec.max()),
             "| Total speech after trimming | %.2f h |" % (df.sec.sum() / 3600), "",
             "| Split | Words | Utterances |", "|---|---|---|"]
    for s in ("train", "val", "test"):
        d = df[df.split == s]
        lines.append("| %s | %d | %d |" % (s, d.word.nunique(), len(d)))
    lines += ["", "Utterances per emotion: " + ", ".join("%s %d" % kv for kv in sorted(Counter(df.emotion).items()))]
    if notes:
        lines += ["", "Irregular files:", ""] + ["- " + n for n in notes]
    os.makedirs(RESULTS, exist_ok=True)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS, "data_summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
