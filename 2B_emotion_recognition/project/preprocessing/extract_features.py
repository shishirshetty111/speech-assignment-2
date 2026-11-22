"""Feature extraction block (both modalities). Run after prepare_data.py.

    python preprocessing/extract_features.py

Speech  (time_steps x features)
  - WavLM-base-plus (microsoft/wavlm-base-plus), a speech model pretrained on 94,000 hours of
    English speech, used frozen. One 768-dimensional vector per 20 ms of audio.
    WavLM has 13 hidden layers (the CNN output and 12 Transformer layers), and emotion information
    is not equally strong in all of them. The layer is chosen with a quick probe: for every layer
    the frames are averaged per utterance and a logistic regression is trained on train words and
    scored on validation words, across speakers (see probe_layers). The test split is not used.
    `--reuse_pooled` reuses the saved layer averages instead of running WavLM twice.
    -> features/speech_wavlm.pt         {file: float16 tensor [frames, 768]} for the chosen layer
       Results/wavlm_layer_probe.csv    validation accuracy of every layer
  - Log-mel spectrogram (64 mel bands, 25 ms window, 10 ms step), a simple hand-crafted baseline.
    -> features/speech_logmel.pt        {file: float32 tensor [frames, 64]}
Text    (tokens x features)
  - BERT-base-uncased, used frozen; the last hidden layer for every WordPiece token of the
    transcript, including [CLS] and [SEP].
    -> features/text_bert.pt            {file: float32 tensor [tokens, 768]}, plus the token strings
"""
import os
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.common import DATA, FEATURES, RESULTS, SR  # noqa: E402

WAVLM = "microsoft/wavlm-base-plus"
BERT = "bert-base-uncased"
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))


def load(file):
    y, sr = sf.read(os.path.join(DATA, "processed", file), dtype="float32")
    assert sr == SR
    return y


class WavLMExtractor:
    def __init__(self):
        self.fe = AutoFeatureExtractor.from_pretrained(WAVLM)
        self.model = AutoModel.from_pretrained(WAVLM).eval()

    @torch.no_grad()
    def __call__(self, y, layer=None):
        x = self.fe(y, sampling_rate=SR, return_tensors="pt").input_values
        hs = self.model(x, output_hidden_states=True).hidden_states  # 13 x [1, frames, 768]
        if layer is None:
            return torch.stack([h[0] for h in hs])  # [13, frames, 768]
        return hs[layer][0]


def logmel(y):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=400, hop_length=160, n_mels=64)
    return torch.from_numpy(np.log(m + 1e-6).T.astype(np.float32))  # [frames, 64]


def _probe(X, y, tr, va):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X[tr]), y[tr])
    return clf.score(sc.transform(X[va]), y[va])


def probe_layers(df, pooled):
    """Two probes per layer, both on train words -> validation words (test words are not used):
      same speakers : train and validate on both speakers. On TESS almost every layer reaches ~100%,
                      so this cannot tell the layers apart.
      cross-speaker : train on one speaker, validate on the other (both directions, averaged). This
                      measures emotion information that is not tied to one voice, and decides the layer."""
    y = df.emotion.values
    tr, va = (df.split == "train").values, (df.split == "val").values
    rows = []
    for layer in range(pooled.shape[1]):
        X = pooled[:, layer]
        same = _probe(X, y, tr, va)
        cross = np.mean([_probe(X, y, tr & (df.speaker == a).values, va & (df.speaker == b).values)
                         for a, b in (("OAF", "YAF"), ("YAF", "OAF"))])
        rows.append(dict(layer=layer, val_acc_same_speakers=round(same, 4), val_acc_cross_speaker=round(cross, 4)))
        print("layer %2d  same speakers %.3f  cross-speaker %.3f" % (layer, same, cross), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, "wavlm_layer_probe.csv"), index=False)
    return int(res.sort_values(["val_acc_cross_speaker", "layer"], ascending=[False, True]).iloc[0].layer)


def main():
    os.makedirs(FEATURES, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA, "manifest.csv"))

    # text
    tok = AutoTokenizer.from_pretrained(BERT)
    bert = AutoModel.from_pretrained(BERT).eval()
    text = {}
    with torch.no_grad():
        for f, t in zip(df.file, df.text):
            enc = tok(t, return_tensors="pt")
            text[f] = bert(**enc).last_hidden_state[0].clone()
    tokens = {t: tok.convert_ids_to_tokens(tok(t).input_ids) for t in df.text.unique()}
    torch.save({"features": text, "tokens": tokens, "model": BERT}, os.path.join(FEATURES, "text_bert.pt"))
    print("text features:", len(text), "example", df.text[0], tokens[df.text[0]])

    # log-mel
    mel = {f: logmel(load(f)) for f in df.file}
    torch.save({"features": mel}, os.path.join(FEATURES, "speech_logmel.pt"))
    print("log-mel features:", len(mel))

    # WavLM: pass 1 pools every layer for the probe, pass 2 keeps the chosen layer
    ext = WavLMExtractor()
    pooled_path = os.path.join(FEATURES, "wavlm_layer_means.npy")
    if "--reuse_pooled" in sys.argv and os.path.exists(pooled_path):
        pooled = np.load(pooled_path)
    else:
        pooled = np.zeros((len(df), 13, 768), dtype=np.float32)
        for i, f in enumerate(df.file):
            pooled[i] = ext(load(f)).mean(dim=1).numpy()
            if i % 200 == 0:
                print("WavLM pass 1: %d/%d" % (i, len(df)), flush=True)
        np.save(pooled_path, pooled)
    layer = probe_layers(df, pooled)
    print("chosen WavLM layer:", layer)
    seq = {}
    for i, f in enumerate(df.file):
        seq[f] = ext(load(f), layer).to(torch.float16).clone()
        if i % 200 == 0:
            print("WavLM pass 2: %d/%d" % (i, len(df)), flush=True)
    torch.save({"features": seq, "layer": layer, "model": WAVLM}, os.path.join(FEATURES, "speech_wavlm.pt"))
    print("done")


if __name__ == "__main__":
    main()
