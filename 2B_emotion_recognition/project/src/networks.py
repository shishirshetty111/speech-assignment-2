"""The trainable blocks: temporal modelling (speech), contextual modelling (text), fusion, classifier.

Every model exposes `represent(batch)`, which returns the output of the block named in the brief
(temporal / contextual / fusion). These are the vectors used for the t-SNE plots.
"""
import torch
import torch.nn as nn

HID = 128          # size of each LSTM direction / Transformer width
REP = 2 * HID      # size of every block's output representation (256)


class AttentionPool(nn.Module):
    """Weighted average over time: a small layer scores each step, softmax turns the scores into
    weights, padded steps get weight 0. Lets the model focus on the emotional part of the clip."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x, mask):  # x [B, T, D], mask [B, T] True = real step
        s = self.score(x).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=1)
        return (w.unsqueeze(-1) * x).sum(1), w


class TemporalSpeech(nn.Module):
    """Speech temporal modelling: 2-layer bidirectional LSTM over the frame features + attention pooling."""

    def __init__(self, d_in, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, HID), nn.GELU())
        self.lstm = nn.LSTM(HID, HID, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttentionPool(REP)

    def forward(self, x, mask):
        h = self.proj(x)
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        rep, self.last_weights = self.pool(out, mask)
        return rep


class ContextualText(nn.Module):
    """Text contextual modelling: 2 Transformer encoder layers (self-attention across tokens) on top of
    the frozen BERT token vectors, then attention pooling."""

    def __init__(self, d_in, dropout=0.3, max_len=32):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, REP))
        self.pos = nn.Parameter(torch.zeros(1, max_len, REP))
        layer = nn.TransformerEncoderLayer(REP, nhead=4, dim_feedforward=2 * REP, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.pool = AttentionPool(REP)

    def forward(self, x, mask):
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        h = self.enc(h, src_key_padding_mask=~mask)
        rep, _ = self.pool(h, mask)
        return rep


class Classifier(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(REP, HID), nn.GELU(), nn.Dropout(dropout), nn.Linear(HID, n_classes))

    def forward(self, r):
        return self.net(r)


class SpeechModel(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.temporal, self.classifier = TemporalSpeech(d_in), Classifier(n_classes)

    def represent(self, b):
        return self.temporal(b["speech"], b["speech_mask"])

    def forward(self, b):
        return self.classifier(self.represent(b))


class TextModel(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.contextual, self.classifier = ContextualText(d_in), Classifier(n_classes)

    def represent(self, b):
        return self.contextual(b["text"], b["text_mask"])

    def forward(self, b):
        return self.classifier(self.represent(b))


class Fusion(nn.Module):
    """'concat': concatenate the two vectors and project back to 256.
    'gated' : a learned gate g in (0, 1) per dimension decides how much of each modality to keep:
              fused = g * speech + (1 - g) * text. The mean gate shows how much the model trusts speech."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.net = nn.Sequential(nn.Linear(2 * REP, REP), nn.GELU())
        else:
            self.gate = nn.Linear(2 * REP, REP)
            self.s_proj, self.t_proj = nn.Linear(REP, REP), nn.Linear(REP, REP)

    def forward(self, s, t):
        st = torch.cat([s, t], dim=-1)
        if self.mode == "concat":
            return self.net(st)
        g = torch.sigmoid(self.gate(st))
        self.last_gate = g
        return g * torch.tanh(self.s_proj(s)) + (1 - g) * torch.tanh(self.t_proj(t))


class FusionModel(nn.Module):
    def __init__(self, d_speech, d_text, n_classes, mode="gated"):
        super().__init__()
        self.temporal, self.contextual = TemporalSpeech(d_speech), ContextualText(d_text)
        self.fusion, self.classifier = Fusion(mode), Classifier(n_classes)

    def represent(self, b):
        return self.fusion(self.temporal(b["speech"], b["speech_mask"]), self.contextual(b["text"], b["text_mask"]))

    def forward(self, b):
        return self.classifier(self.represent(b))
