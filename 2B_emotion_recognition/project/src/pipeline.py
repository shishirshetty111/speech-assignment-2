"""Training and testing shared by the three pipelines (speech, text, fusion).

Variants
  speech_wavlm   WavLM features  -> BiLSTM + attention (temporal)              -> classifier   main speech model
  speech_logmel  log-mel         -> same temporal block                        -> classifier   simple baseline
  text_bert      BERT tokens     -> 2-layer Transformer + attention (contextual)-> classifier
  fusion_concat  speech_wavlm and text_bert blocks -> concatenation fusion      -> classifier
  fusion_gated   speech_wavlm and text_bert blocks -> gated fusion              -> classifier

Every variant is trained 5 times with different random seeds (0-4); results are reported as
mean and standard deviation over the 5 runs. Training uses the train split; the checkpoint with the
best validation accuracy is kept (early stopping); the test split is only used by test().
"""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from src.common import CHECKPOINTS, DATA, EMOTIONS, FEATURES
from src.networks import FusionModel, SpeechModel, TextModel

VARIANTS = {
    "speech_wavlm": dict(kind="speech", speech="wavlm"),
    "speech_logmel": dict(kind="speech", speech="logmel"),
    "text_bert": dict(kind="text"),
    "fusion_concat": dict(kind="fusion", speech="wavlm", mode="concat"),
    "fusion_gated": dict(kind="fusion", speech="wavlm", mode="gated"),
}
SEEDS = [0, 1, 2, 3, 4]
BATCH, LR, WD, MAX_EPOCHS, PATIENCE = 32, 1e-3, 1e-2, 60, 12
LABEL = {e: i for i, e in enumerate(EMOTIONS)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_cache = {}


def manifest():
    return pd.read_csv(os.path.join(DATA, "manifest.csv"))


def features(name):
    if name not in _cache:
        path = {"wavlm": "speech_wavlm.pt", "logmel": "speech_logmel.pt", "text": "text_bert.pt"}[name]
        _cache[name] = torch.load(os.path.join(FEATURES, path), weights_only=False)["features"]
    return _cache[name]


class Batches:
    """Pads variable-length sequences in each batch and builds masks (True = real step)."""

    def __init__(self, df, cfg, norm=None, shuffle=False, seed=0, speech_override=None, text_override=None):
        self.df, self.cfg, self.norm, self.shuffle = df.reset_index(drop=True), cfg, norm, shuffle
        self.rng = random.Random(seed)
        self.speech = speech_override or (features(cfg["speech"]) if "speech" in cfg else None)
        self.text = (text_override or features("text")) if cfg["kind"] in ("text", "fusion") else None

    def __iter__(self):
        idx = list(range(len(self.df)))
        if self.shuffle:
            self.rng.shuffle(idx)
        for i in range(0, len(idx), BATCH):
            rows = self.df.iloc[idx[i: i + BATCH]]
            b = {"y": torch.tensor([LABEL[e] for e in rows.emotion]), "file": list(rows.file)}
            if self.speech is not None:
                seqs = [self.speech[f].float() for f in rows.file]
                if self.norm is not None:
                    seqs = [(s - self.norm[0]) / self.norm[1] for s in seqs]
                b["speech"], b["speech_mask"] = _pad(seqs)
            if self.text is not None:
                b["text"], b["text_mask"] = _pad([self.text[f] for f in rows.file])
            yield {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in b.items()}


def _pad(seqs):
    lengths = torch.tensor([len(s) for s in seqs])
    x = pad_sequence(seqs, batch_first=True)
    return x, torch.arange(x.shape[1])[None, :] < lengths[:, None]


def speech_norm(cfg, train_files):
    """Per-dimension mean and std of the log-mel features, from the training files only.
    WavLM features go through a LayerNorm inside the model instead."""
    if cfg.get("speech") != "logmel":
        return None
    x = torch.cat([features("logmel")[f] for f in train_files])
    return x.mean(0), x.std(0) + 1e-5


def build(cfg):
    d_speech = {"wavlm": 768, "logmel": 64}.get(cfg.get("speech"))
    if cfg["kind"] == "speech":
        return SpeechModel(d_speech, len(EMOTIONS))
    if cfg["kind"] == "text":
        return TextModel(768, len(EMOTIONS))
    return FusionModel(d_speech, 768, len(EMOTIONS), cfg["mode"])


@torch.no_grad()
def run_eval(model, batches):
    model.eval()
    ys, ps, probs, reps, files = [], [], [], [], []
    loss, n = 0.0, 0
    for b in batches:
        r = model.represent(b)
        logits = model.classifier(r)
        loss += nn.functional.cross_entropy(logits, b["y"], reduction="sum").item()
        n += len(b["y"])
        ys.append(b["y"].cpu())
        ps.append(logits.argmax(-1).cpu())
        probs.append(torch.softmax(logits, -1).cpu())
        reps.append(r.cpu())
        files += b["file"]
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return dict(acc=float((y == p).mean()), loss=loss / n, y=y, pred=p, prob=torch.cat(probs).numpy(),
                rep=torch.cat(reps).numpy(), file=files)


def ckpt_path(variant, seed, tag=""):
    return os.path.join(CHECKPOINTS, "%s%s_seed%d.pt" % (variant, tag, seed))


def train(variant, seed, df=None, tag="", log=print):
    """Train one variant with one seed. df must have a 'split' column (train / val / test)."""
    cfg = VARIANTS[variant]
    df = manifest() if df is None else df
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    norm = speech_norm(cfg, tr.file)
    model = build(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    best, best_state, wait, history = (-1, 1e9), None, 0, []
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tl, tn, tc = 0.0, 0, 0
        for b in Batches(tr, cfg, norm, shuffle=True, seed=seed * 1000 + epoch):
            logits = model(b)
            loss = nn.functional.cross_entropy(logits, b["y"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(b["y"])
            tn += len(b["y"])
            tc += (logits.argmax(-1) == b["y"]).sum().item()
        v = run_eval(model, Batches(va, cfg, norm))
        sched.step(v["acc"])
        history.append(dict(epoch=epoch, train_loss=tl / tn, train_acc=tc / tn, val_loss=v["loss"], val_acc=v["acc"]))
        if (v["acc"], -v["loss"]) > (best[0], -best[1]):
            best, wait = (v["acc"], v["loss"]), 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    os.makedirs(CHECKPOINTS, exist_ok=True)
    torch.save({"variant": variant, "cfg": cfg, "seed": seed, "state": best_state, "norm": norm,
                "history": history, "best_val_acc": best[0]}, ckpt_path(variant, seed, tag))
    log("%s%s seed %d: best validation accuracy %.3f after %d epochs (%.0f s, %s)"
        % (variant, tag, seed, best[0], len(history), time.time() - t0, DEVICE), flush=True)
    return history


def load(variant, seed, tag=""):
    ck = torch.load(ckpt_path(variant, seed, tag), weights_only=False, map_location="cpu")
    model = build(ck["cfg"])
    model.load_state_dict(ck["state"])
    return model.to(DEVICE).eval(), ck


def evaluate(variant, seed, df, split="test", tag="", speech_override=None, text_override=None):
    model, ck = load(variant, seed, tag)
    return run_eval(model, Batches(df[df.split == split], ck["cfg"], ck["norm"], speech_override=speech_override,
                                   text_override=text_override))


def save_json(obj, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
