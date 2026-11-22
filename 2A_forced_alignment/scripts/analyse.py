"""Step 5: numbers for the report, from the TextGrids in outputs/textgrids/.

  outputs/analysis/experiment_summary.csv   per experiment: words, phones, unknown-word (spn) phones,
                                            silences, phone durations, MFA speech log-likelihood
  outputs/analysis/boundary_agreement.csv   word and phone boundary differences of every experiment
                                            against the main result (arpa_g2p)
  outputs/analysis/short_phones.csv         phones at the 30 ms minimum MFA allows (often squeezed)
  outputs/analysis/isle_vowels.csv          learner minimal pairs: vowel duration and F1/F2 (Praat)
  outputs/analysis/summary.md               the same tables in one readable file

Agreement between two aligners is not accuracy: there are no hand-marked reference boundaries.
"""
import csv
import difflib
import glob
import os
import subprocess
import sys

import numpy as np
from praatio import textgrid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANALYSIS, CORPUS, ROOT, TEXTGRIDS, praat_executable  # noqa: E402

REFERENCE = "arpa_g2p"
SILENCE = {"", "sil", "sp"}
PRAAT = praat_executable()
ARPA = {"arpa_baseline", "arpa_g2p", "arpa_own_g2p", "arpa_adapted", "scratch"}
# the target word in each ISLE prompt and its contrast word: "I SAID <a> NOT <b>"
ISLE_WORDS = ["white", "bait", "new", "no", "bad", "bed"]


def experiments():
    return sorted(d for d in os.listdir(TEXTGRIDS) if os.path.isdir(os.path.join(TEXTGRIDS, d)))


def load(exp):
    """utt -> {"words": [(start, end, label)], "phones": [...]} (silences removed)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(TEXTGRIDS, exp, "*", "*.TextGrid"))):
        tg = textgrid.openTextgrid(p, includeEmptyIntervals=False)
        utt = os.path.splitext(os.path.basename(p))[0]
        out[utt] = {t: [(e.start, e.end, e.label) for e in tg.getTier(t).entries if e.label not in SILENCE]
                    for t in ("words", "phones")}
        w = out[utt]["words"]  # MFA 3 leaves silences unlabelled: pauses are the gaps between words
        out[utt]["sil"] = [(a[1], b[0]) for a, b in zip(w, w[1:]) if b[0] - a[1] > 1e-6]
    return out


def likelihoods(exp):
    p = os.path.join(TEXTGRIDS, exp, "alignment_analysis.csv")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return {r["file"]: float(r["speech_log_likelihood"]) for r in csv.DictReader(f)}


def boundary_diffs(ref, other):
    """Absolute start/end differences (s) of matching items (same label in the same order)."""
    diffs = []
    sm = difflib.SequenceMatcher(a=[x[2] for x in ref], b=[x[2] for x in other], autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            r, o = ref[a + k], other[b + k]
            diffs += [abs(r[0] - o[0]), abs(r[1] - o[1])]
    return diffs, sum(n for _, _, n in sm.get_matching_blocks()), len(ref)


def praat(wav, start, end, max_formant=5500):
    res = subprocess.run([PRAAT, "--run", os.path.join(ROOT, "scripts", "measure_vowel.praat"), wav,
                          "%.4f" % start, "%.4f" % end, str(max_formant)], capture_output=True)
    out = res.stdout
    # Praat on Windows writes UTF-16
    return (out.decode("utf-16") if b"\x00" in out else out.decode("utf-8", "replace")).split()


def formant_ceiling(wav):
    """Praat's usual advice: 5000 Hz for a male voice, 5500 Hz for a female voice; the speaker's
    sex is not given, so it is inferred from the median F0 (below 165 Hz -> 5000 Hz)."""
    f0 = float(praat(wav, 0, 0)[0])
    return f0, (5000 if f0 < 165 else 5500)


def write_csv(name, rows):
    with open(os.path.join(ANALYSIS, name), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md_table(rows):
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    data = {e: load(e) for e in experiments()}
    if REFERENCE not in data:
        sys.exit("run the %s alignment first" % REFERENCE)

    summary, short = [], []
    for exp, utts in data.items():
        ph = [p for u in utts.values() for p in u["phones"]]
        dur = np.array([e - s for s, e, _ in ph])
        ll = likelihoods(exp)
        summary.append({
            "experiment": exp,
            "words": sum(len(u["words"]) for u in utts.values()),
            "phones": len(ph),
            "spn_phones": sum(lab == "spn" for _, _, lab in ph),
            "pauses": sum(len(u["sil"]) for u in utts.values()),
            "pause_s": round(sum(e - s for u in utts.values() for s, e in u["sil"]), 2),
            "mean_phone_ms": round(1000 * dur.mean(), 1),
            "phones_at_30ms": int((dur <= 0.0301).sum()),
            "speech_loglik_f2b": round(np.mean([v for k, v in ll.items() if k.startswith("F2B")]), 2) if ll else "",
            "speech_loglik_isle": round(np.mean([v for k, v in ll.items() if k.startswith("ISLE")]), 2) if ll else "",
        })
        if exp == REFERENCE:
            for utt, u in utts.items():
                for s, e, lab in u["phones"]:
                    if e - s <= 0.0301:
                        word = next((w for ws, we, w in u["words"] if ws <= s < we), "")
                        short.append({"file": utt, "word": word, "phone": lab, "start": round(s, 3),
                                      "ms": round(1000 * (e - s))})

    agree = []
    for exp in data:
        if exp == REFERENCE:
            continue
        for tier in ("words", "phones"):
            if tier == "phones" and (exp in ARPA) != (REFERENCE in ARPA):
                continue  # different phone sets cannot be matched phone by phone
            diffs, matched, total = [], 0, 0
            for utt, ref in data[REFERENCE].items():
                d, m, t = boundary_diffs(ref[tier], data[exp][utt][tier])
                diffs += d
                matched += m
                total += t
            diffs = np.array(diffs) * 1000
            agree.append({"experiment": exp, "tier": tier, "matched": "%d/%d" % (matched, total),
                          "mean_ms": round(diffs.mean(), 1), "median_ms": round(float(np.median(diffs)), 1),
                          "within_20ms_pct": round(100 * (diffs <= 20).mean(), 1),
                          "max_ms": round(diffs.max())})

    vowels, ceilings = [], {}
    for exp in sorted(data):
        for utt, u in data[exp].items():
            if not utt.startswith("ISLE"):
                continue
            wav = glob.glob(os.path.join(CORPUS, "*", utt + ".wav"))[0]
            if wav not in ceilings:
                ceilings[wav] = formant_ceiling(wav)
            f0, ceiling = ceilings[wav]
            for ws, we, w in u["words"]:
                if w not in ISLE_WORDS:
                    continue
                inside = [p for p in u["phones"] if p[0] >= ws - 1e-6 and p[1] <= we + 1e-6]
                vowel = [p for p in inside if p[2][0].lower() in "aeiouɐɛɪʊəæɑɔ" or p[2][-1].isdigit()]
                if not vowel:
                    continue
                s, e, lab = vowel[0]
                vals = praat(wav, s, e, ceiling)
                f1, f2 = vals if len(vals) == 2 else ("", "")
                vowels.append({"experiment": exp, "file": utt[-9:-6], "median_F0_Hz": round(f0),
                               "ceiling_Hz": ceiling, "word": w,
                               "phones": " ".join(p[2] for p in inside), "vowel": lab,
                               "vowel_ms": round(1000 * (e - s)), "F1_Hz": f1, "F2_Hz": f2})

    write_csv("experiment_summary.csv", summary)
    write_csv("boundary_agreement.csv", agree)
    write_csv("short_phones.csv", short or [{"file": "", "word": "", "phone": "", "start": "", "ms": ""}])
    write_csv("isle_vowels.csv", vowels)
    with open(os.path.join(ANALYSIS, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# Alignment analysis\n\nReference for agreement: %s. Agreement is not accuracy "
                "(no hand-marked boundaries exist for this data).\n\n" % REFERENCE)
        f.write("## Experiments\n\n" + md_table(summary) + "\n\n")
        f.write("## Boundary agreement with %s\n\n" % REFERENCE + md_table(agree) + "\n\n")
        by_phone = {}
        for r in short:
            by_phone[r["phone"]] = by_phone.get(r["phone"], 0) + 1
        common_ = ", ".join("%s %d" % kv for kv in sorted(by_phone.items(), key=lambda kv: -kv[1])[:8])
        f.write("## Phones at the 30 ms minimum (%s)\n\n%d phones (full list in short_phones.csv). "
                "Most common: %s\n\n" % (REFERENCE, len(short), common_))
        f.write("## ISLE minimal pairs: vowel duration and formants (%s; all experiments in isle_vowels.csv)\n\n"
                % REFERENCE + md_table([v for v in vowels if v["experiment"] == REFERENCE]) + "\n")
    print(open(os.path.join(ANALYSIS, "summary.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
