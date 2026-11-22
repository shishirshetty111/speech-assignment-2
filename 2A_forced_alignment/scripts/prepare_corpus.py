"""Step 1: put the provided data into the layout MFA expects.

The BU Radio News files (F2B*) are NIST SPHERE audio with a .wav name, not RIFF WAV.
They are converted to standard 16-bit PCM WAV (samples unchanged); RIFF files are copied.

    data/raw/wav/<id>.wav + data/raw/transcripts/<id>.txt|.TXT
 -> data/corpus/<speaker>/<id>.wav + <id>.lab

The original transcripts are not changed. The .lab copy only gets mechanical clean-up,
and every change is written to outputs/analysis/text_changes.tsv:
  - line breaks and repeated spaces are collapsed;
  - a hyphen followed by a space inside a number word ("seventy- six") is joined ("seventy-six");
  - extra rules for abbreviations are listed in NORMALISE below, each with its reason.
No spelling is corrected.
"""
import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANALYSIS, CORPUS, RAW, speaker_of  # noqa: E402

import wave  # noqa: E402

import numpy as np  # noqa: E402


def read_audio(path):
    """Return (int16 samples, sample rate, format name) for a RIFF WAV or NIST SPHERE file."""
    with open(path, "rb") as f:
        head = f.read(1024)
    if head.startswith(b"RIFF"):
        with wave.open(path) as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                sys.exit("%s: expected 16-bit mono" % path)
            return np.frombuffer(w.readframes(w.getnframes()), "<i2"), w.getframerate(), "RIFF WAV"
    if head.startswith(b"NIST_1A"):
        hdr_len = int(head.split()[1])
        with open(path, "rb") as f:
            hdr = f.read(hdr_len).decode("ascii", "replace")
            fields = {}
            for line in hdr.splitlines()[2:]:
                parts = line.split(None, 2)
                if len(parts) == 3:
                    fields[parts[0]] = parts[2]
            if fields.get("sample_coding", "pcm") != "pcm" or fields.get("sample_n_bytes") != "2" \
                    or fields.get("channel_count") != "1":
                sys.exit("%s: unsupported SPHERE coding %s" % (path, fields))
            order = "<i2" if fields.get("sample_byte_format", "01") == "01" else ">i2"
            x = np.frombuffer(f.read(), order)
        if len(x) != int(fields["sample_count"]):
            sys.exit("%s: sample_count mismatch" % path)
        return x.astype("<i2"), int(fields["sample_rate"]), "NIST SPHERE"
    sys.exit("%s: unknown audio format" % path)


def write_wav(path, x, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())

# (pattern, replacement, reason). Applied in order to the joined transcript.
NORMALISE = [
    (r"(\w+)- (\w+)",r"\1-\2", "hyphen split across a space (line-break artefact)"),
]


def clean(text):
    changes = []
    joined = re.sub(r"\s+", " ", text).strip()
    if joined != text:
        changes.append(("whitespace", "line breaks / repeated spaces collapsed"))
    for pat, rep, why in NORMALISE:
        for m in re.finditer(pat, joined):
            changes.append((m.group(0), m.expand(rep) + "  (" + why + ")"))
        joined = re.sub(pat, rep, joined)
    return joined, changes


def main():
    if os.path.isdir(CORPUS):
        shutil.rmtree(CORPUS)
    os.makedirs(ANALYSIS, exist_ok=True)
    log = ["file\tfrom\tto"]
    wavs = sorted(glob.glob(os.path.join(RAW, "wav", "*.wav")))
    if not wavs:
        sys.exit("no wav files in %s (see README: put the provided data there)" % RAW)
    for wav in wavs:
        utt = os.path.splitext(os.path.basename(wav))[0]
        txt = [p for p in glob.glob(os.path.join(RAW, "transcripts", utt + ".*"))]
        if len(txt) != 1:
            sys.exit("expected one transcript for %s, found %s" % (utt, txt))
        x, sr, fmt = read_audio(wav)
        with open(txt[0], encoding="utf-8") as f:
            text, changes = clean(f.read())
        spk_dir = os.path.join(CORPUS, speaker_of(utt))
        os.makedirs(spk_dir, exist_ok=True)
        write_wav(os.path.join(spk_dir, utt + ".wav"), x, sr)
        with open(os.path.join(spk_dir, utt + ".lab"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        log += ["%s\t%s\t%s" % (utt, a, b) for a, b in changes]
        print("%-36s %-14s %-11s %5.2f s  %d Hz  %d words" % (utt, speaker_of(utt), fmt, len(x) / sr, sr,
                                                             len(text.split())))
    with open(os.path.join(ANALYSIS, "text_changes.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(log) + "\n")
    print("corpus written to", CORPUS)


if __name__ == "__main__":
    main()
