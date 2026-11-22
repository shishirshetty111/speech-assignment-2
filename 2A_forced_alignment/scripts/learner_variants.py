"""Extra experiment: let the aligner choose between the expected and a learner pronunciation.

The ISLE prompts are minimal pairs ("I SAID <A> NOT <B>"). For each pair the vowel of the other
word is added as an alternative pronunciation, in both directions, so the test is symmetric:
    bad  : B AE1 D  or  B EH1 D        bed  : B EH1 D  or  B AE1 D
    white: W AY1 T  or  W EY1 T        bait : B EY1 T  or  B AY1 T
    new  : N UW1 / N Y UW1  or  N OW1  no   : N OW1  or  N UW1
All pronunciations of these words get the same prior probability (1.0), so the choice is made by
the acoustic model alone. MFA picks, for each word, the pronunciation that fits the audio best.

    python scripts/learner_variants.py     (needs work/dicts/arpa_pretrained_g2p.dict)
writes dictionaries/arpa_learner_variants_additions.dict, work/dicts/arpa_learner_variants.dict,
aligns the corpus (experiment arpa_learner) and writes outputs/analysis/learner_variants.md
"""
import glob
import os
import subprocess
import sys

from praatio import textgrid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANALYSIS, DICTS, TEXTGRIDS, WORK  # noqa: E402

PAIRS = [("white", "bait", "AY1", "EY1"), ("new", "no", "UW1", "OW1"), ("bad", "bed", "AE1", "EH1")]
BASE = os.path.join(WORK, "dicts", "arpa_pretrained_g2p.dict")
OUT = os.path.join(WORK, "dicts", "arpa_learner_variants.dict")


def build():
    words = {w for a, b, _, _ in PAIRS for w in (a, b)}
    swap = {}
    for a, b, va, vb in PAIRS:
        swap[a], swap[b] = (va, vb), (vb, va)
    kept, entries = [], {}
    with open(BASE, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in words:
                entries.setdefault(parts[0], []).append(parts)
            else:
                kept.append(line)
    new_lines, additions = [], []
    for w in sorted(words):
        own, other = swap[w]
        prons = []
        for parts in entries[w]:
            phones = parts[-1]
            if len(phones.split()) < 2 or not any(p == own for p in phones.split()):
                continue  # drop truncated variants such as "bad: B AE1"
            prons.append((phones, parts))
            alt = " ".join(other if p == own else p for p in phones.split())
            prons.append((alt, parts))
            additions.append("%s\t%s" % (w, alt))
        seen = set()
        for phones, parts in prons:
            if phones in seen:
                continue
            seen.add(phones)
            sil = parts[2:5] if len(parts) == 6 else ["0.5", "1.0", "1.0"]
            new_lines.append("\t".join([w, "1.0", *sil, phones]) + "\n")
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(kept + new_lines)
    with open(os.path.join(DICTS, "arpa_learner_variants_additions.dict"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(additions) + "\n")
    print("".join(new_lines))


def report():
    rows = ["| file | word | expected vowel | other word's vowel | aligner chose | phones |",
            "|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(TEXTGRIDS, "arpa_learner", "*", "*.TextGrid"))):
        tg = textgrid.openTextgrid(p, includeEmptyIntervals=False)
        phones = tg.getTier("phones").entries
        for e in tg.getTier("words").entries:
            for a, b, va, vb in PAIRS:
                if e.label not in (a, b):
                    continue
                own, other = (va, vb) if e.label == a else (vb, va)
                ph = [x.label for x in phones if x.start >= e.start - 1e-6 and x.end <= e.end + 1e-6]
                chose = "expected" if own in ph else ("other word's vowel" if other in ph else "?")
                rows.append("| %s | %s | %s | %s | %s | %s |" % (os.path.basename(p).split("_")[3] if "ISLE" in p else os.path.basename(p)[:-9], e.label, own,
                                                                  other, chose, " ".join(ph)))
    text = ("# Learner pronunciation variants\n\nEach target word could be aligned with its own vowel or "
            "with the vowel of the other word in the pair (equal prior probability).\n\n" + "\n".join(rows) + "\n")
    with open(os.path.join(ANALYSIS, "learner_variants.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


def main():
    build()
    res = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "align_all.py"),
                          "--only", "arpa_learner"])
    if res.returncode:
        sys.exit("alignment failed")
    report()


if __name__ == "__main__":
    main()
