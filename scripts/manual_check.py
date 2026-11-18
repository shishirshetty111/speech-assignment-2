"""Manual checks of the alignment: hand-marked word boundaries and the added OOV pronunciations.

    python scripts/manual_check.py prepare   # blank TextGrids to mark by hand + OOV audio clips
    python scripts/manual_check.py score     # compare the hand-marked boundaries with each model

prepare
  outputs/manual/<file>.TextGrid   one empty "words" tier per ISLE file. The boundaries are marked by
                                   hand in Praat from the audio only (the MFA result is not shown, so
                                   the hand marking is not biased towards it).
  work/oov_clips/<word>.wav        each word that was missing from the dictionary, cut from the
                                   audio with 0.15 s either side, to listen to (not committed: the
                                   audio is licensed). The phones aligned for it are printed.
score
  outputs/analysis/manual_check.md  for every experiment: mean and median boundary difference and the
                                    share of boundaries within 20 ms and 50 ms of the hand marking.
"""
import glob
import os
import statistics
import sys
import wave

from praatio import textgrid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANALYSIS, CORPUS, OUT, TEXTGRIDS, WORK  # noqa: E402

MANUAL = os.path.join(OUT, "manual")
FILES = ["ISLE_SESS0131_BLOCKD02_01_sprt1", "ISLE_SESS0131_BLOCKD02_02_sprt1", "ISLE_SESS0131_BLOCKD02_03_sprt1"]
OOVS = ["wbur's", "melnicove", "dukakis", "maffy", "politicize"]
MAIN = "arpa_g2p"


def wav_of(utt):
    return glob.glob(os.path.join(CORPUS, "*", utt + ".wav"))[0]


def tg_of(exp, utt):
    found = glob.glob(os.path.join(TEXTGRIDS, exp, "*", utt + ".TextGrid"))
    return found[0] if found else None


def duration(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def prepare():
    os.makedirs(MANUAL, exist_ok=True)
    for utt in FILES:
        path = os.path.join(MANUAL, utt + ".TextGrid")
        if os.path.exists(path):
            print("kept (already exists):", path)
            continue
        tg = textgrid.Textgrid()
        tg.addTier(textgrid.IntervalTier("words", [], 0, duration(wav_of(utt))))
        tg.save(path, format="long_textgrid", includeBlankSpaces=True)
        print("blank TextGrid:", path)

    clips = os.path.join(WORK, "oov_clips")
    os.makedirs(clips, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(TEXTGRIDS, MAIN, "*", "*.TextGrid"))):
        tg = textgrid.openTextgrid(p, includeEmptyIntervals=False)
        phones = tg.getTier("phones").entries
        utt = os.path.splitext(os.path.basename(p))[0]
        for e in tg.getTier("words").entries:
            if e.label not in OOVS:
                continue
            ph = " ".join(x.label for x in phones if x.start >= e.start - 1e-6 and x.end <= e.end + 1e-6)
            out = os.path.join(clips, "%s_%s.wav" % (e.label.replace("'", ""), utt))
            with wave.open(wav_of(utt)) as w:
                rate = w.getframerate()
                a = max(0, int((e.start - 0.15) * rate))
                b = min(w.getnframes(), int((e.end + 0.15) * rate))
                w.setpos(a)
                frames = w.readframes(b - a)
                params = w.getparams()
            with wave.open(out, "wb") as o:
                o.setparams(params)
                o.writeframes(frames)
            print("%-11s %s %.2f-%.2f s  %s\n            %s" % (e.label, utt, e.start, e.end, ph, out))


def boundaries(entries):
    """Word start and end times, in order, ignoring empty intervals."""
    words = [e for e in entries if e.label.strip()]
    return [(e.label.strip().lower(), e.start, e.end) for e in words]


def score():
    exps = sorted(d for d in os.listdir(TEXTGRIDS) if os.path.isdir(os.path.join(TEXTGRIDS, d)))
    diffs = {e: [] for e in exps}
    marked = []
    for utt in FILES:
        path = os.path.join(MANUAL, utt + ".TextGrid")
        ref = boundaries(textgrid.openTextgrid(path, includeEmptyIntervals=False).getTier("words").entries)
        if not ref:
            print("no words marked yet in", path)
            continue
        marked.append((utt, len(ref)))
        for exp in exps:
            p = tg_of(exp, utt)
            if not p:
                continue
            hyp = boundaries(textgrid.openTextgrid(p, includeEmptyIntervals=False).getTier("words").entries)
            if [w for w, _, _ in hyp] != [w for w, _, _ in ref]:
                sys.exit("%s: hand-marked words %s do not match %s words %s"
                         % (utt, [w for w, _, _ in ref], exp, [w for w, _, _ in hyp]))
            for (_, rs, re_), (_, hs, he) in zip(ref, hyp):
                diffs[exp] += [abs(rs - hs) * 1000, abs(re_ - he) * 1000]
    if not marked:
        sys.exit("nothing to score: mark the words in outputs/manual/*.TextGrid first")
    rows = ["| Experiment | Boundaries | Mean difference | Median difference | Within 20 ms | Within 50 ms |",
            "|---|---|---|---|---|---|"]
    for exp in exps:
        d = diffs[exp]
        if not d:
            continue
        rows.append("| %s | %d | %.1f ms | %.1f ms | %.0f%% | %.0f%% |" % (
            exp, len(d), statistics.mean(d), statistics.median(d),
            100 * sum(x <= 20 for x in d) / len(d), 100 * sum(x <= 50 for x in d) / len(d)))
    text = ("# Alignment against hand-marked word boundaries\n\nFiles marked by hand in Praat: %s. "
            "Each word gives two boundaries (start and end).\n\n%s\n"
            % (", ".join("%s (%d words)" % m for m in marked), "\n".join(rows)))
    with open(os.path.join(ANALYSIS, "manual_check.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    {"prepare": prepare, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "prepare"]()
