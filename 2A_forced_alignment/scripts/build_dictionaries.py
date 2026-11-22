"""Step 3: build pronunciation dictionaries that cover every word in the corpus.

For a base MFA dictionary, the words missing from it (out-of-vocabulary, OOV) are taken from
`mfa validate` and given pronunciations:
  - an OOV written in capitals in the transcript (an acronym such as WBUR) is spelled out
    letter by letter, using the dictionary's own pronunciations of the single letters;
  - every other OOV gets the pronunciation predicted by a G2P (grapheme-to-phoneme) model.

    python scripts/build_dictionaries.py --variant arpa_pretrained_g2p
writes dictionaries/<variant>_additions.dict (only the new entries, kept in the repo) and
work/dicts/<variant>.dict (base dictionary + additions, the file given to MFA).
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CORPUS, DICTS, WORK, run_mfa  # noqa: E402

# variant -> (base dictionary, G2P model). Names without a path are MFA pretrained models;
# g2p_own_arpa is trained by scripts/train_g2p.py.
VARIANTS = {
    "arpa_pretrained_g2p": ("english_us_arpa", "english_us_arpa"),
    "arpa_own_g2p": ("english_us_arpa", os.path.join(WORK, "models", "g2p_own_arpa.zip")),
    "mfa_pretrained_g2p": ("english_mfa", "english_us_mfa"),
}


def mfa_root():
    return os.environ.get("MFA_ROOT_DIR", os.path.join(os.path.expanduser("~"), "Documents", "MFA"))


def base_dict_path(name):
    return os.path.join(mfa_root(), "pretrained_models", "dictionary", name + ".dict")


def read_dict(path):
    """word -> list of (probability, phones). Entries without probabilities count as 1.0."""
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            prob = float(parts[1]) if len(parts) == 6 else 1.0
            d.setdefault(parts[0], []).append((prob, parts[-1]))
    return d


def find_oovs(base):
    """Run `mfa validate` (corpus and dictionary checks only) and read its OOV list."""
    tmp = os.path.join(WORK, "validate_" + base)
    run_mfa("validate", CORPUS, base, "--clean", "--ignore_acoustics", "--temporary_directory", tmp,
            "--output_directory", os.path.join(tmp, "report"),
            log=os.path.join(WORK, "logs", "validate_%s.log" % base))
    path = os.path.join(tmp, "report", "oovs_found_%s.txt" % base)
    with open(path, encoding="utf-8") as f:
        return sorted({w.strip() for w in f if w.strip()})


def transcript_text():
    text = ""
    for lab in glob.glob(os.path.join(CORPUS, "*", "*.lab")):
        with open(lab, encoding="utf-8") as f:
            text += f.read() + " "
    return text


def is_acronym(word, text):
    """True if every occurrence of the word in the transcripts is written in capitals while the
    transcript itself is in mixed case (so all-capital transcripts such as ISLE do not count)."""
    hits = re.findall(r"\b%s\b" % re.escape(word), text, flags=re.IGNORECASE)
    lines_mixed = any(c.islower() for c in text)
    return lines_mixed and hits and all(h.isupper() and len(h) > 1 for h in hits)


def spell_out(word, base):
    letters = []
    for ch in word:
        prons = base.get(ch)
        if not prons:
            return None
        letters.append(max(prons, key=lambda p: p[0])[1])  # highest probability, first listed on ties
    return " ".join(letters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    args = ap.parse_args()
    base_name, g2p_model = VARIANTS[args.variant]
    base = read_dict(base_dict_path(base_name))
    oovs = find_oovs(base_name)
    text = transcript_text()
    print("%s: %d OOV words: %s" % (base_name, len(oovs), " ".join(oovs)))

    additions, rows = [], []
    to_g2p = []
    for w in oovs:
        pron = spell_out(w, base) if is_acronym(w, text) else None
        if pron:
            additions.append((w, pron))
            rows.append((w, pron, "acronym: letter pronunciations from " + base_name))
        else:
            to_g2p.append(w)
    if to_g2p:
        os.makedirs(os.path.join(WORK, "g2p"), exist_ok=True)
        words_in = os.path.join(WORK, "g2p", args.variant + "_oovs.txt")
        g2p_out = os.path.join(WORK, "g2p", args.variant + "_g2p.dict")
        with open(words_in, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(to_g2p) + "\n")
        run_mfa("g2p", words_in, g2p_model, g2p_out, "--clean", "--num_pronunciations", "1",
                "--temporary_directory", os.path.join(WORK, "g2p", "tmp_" + args.variant),
                log=os.path.join(WORK, "logs", "g2p_%s.log" % args.variant))
        got = read_dict(g2p_out)
        model_name = os.path.basename(g2p_model)
        for w in to_g2p:
            if w not in got:
                sys.exit("G2P gave no pronunciation for %s" % w)
            additions.append((w, got[w][0][1]))
            rows.append((w, got[w][0][1], "G2P model " + model_name))

    os.makedirs(DICTS, exist_ok=True)
    with open(os.path.join(DICTS, args.variant + "_additions.dict"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join("%s\t%s\n" % a for a in additions))
    with open(os.path.join(DICTS, args.variant + "_additions.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("word\tpronunciation\tsource\n" + "".join("%s\t%s\t%s\n" % r for r in rows))
    os.makedirs(os.path.join(WORK, "dicts"), exist_ok=True)
    full = os.path.join(WORK, "dicts", args.variant + ".dict")
    with open(base_dict_path(base_name), encoding="utf-8") as src, \
            open(full, "w", encoding="utf-8", newline="\n") as dst:
        dst.write(src.read())
        dst.write("".join("%s\t%s\n" % a for a in additions))
    for r in rows:
        print("  %-12s %-40s %s" % r)
    print("wrote", full)


if __name__ == "__main__":
    main()
