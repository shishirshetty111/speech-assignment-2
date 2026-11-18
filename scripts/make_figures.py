"""Step 6: alignment pictures drawn by Praat (waveform, spectrogram, TextGrid tiers).

  outputs/figures/arpa_g2p/<file>.png        every file, main result (whole file)
  outputs/figures/oov_<experiment>.png       F2BJRLP1 19.8-23.0 s: "WBUR's Margo Melnicove"
  outputs/figures/compare_<name>.png         one spectrogram with the phone tier of every
                                             experiment stacked underneath
Praat is found by common.praat_executable() (PRAAT environment variable or PATH).
"""
import glob
import os
import subprocess
import sys

from praatio import textgrid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CORPUS, FIGURES, ROOT, TEXTGRIDS, WORK, praat_executable  # noqa: E402

PRAAT = praat_executable()
MAIN = "arpa_g2p"
DRAW = os.path.join(ROOT, "scripts", "draw_alignment.praat")
ORDER = ["arpa_baseline", "arpa_g2p", "arpa_own_g2p", "arpa_adapted", "scratch", "mfa_g2p"]

# (name, file, start, end, title)
ZOOMS = [
    ("oov", "F2BJRLP1", 19.8, 23.0, "WBUR's Margo Melnicove"),
]
COMPARE = [
    ("isle_bad_bed", "ISLE_SESS0131_BLOCKD02_03_sprt1", 0.3, 3.3, "I SAID BAD NOT BED"),
    ("isle_white_bait", "ISLE_SESS0131_BLOCKD02_01_sprt1", 0.0, 0.0, "I SAID WHITE NOT BAIT"),
    ("isle_new_no", "ISLE_SESS0131_BLOCKD02_02_sprt1", 0.0, 0.0, "I SAID NEW NOT NO"),
    ("f2b_sjc", "F2BJRLP1", 3.9, 6.6, "In April, the S.J.C.'s current leader"),
    ("f2b_breath", "F2BJRLP1", 19.0, 21.2, "...one of the toughest. As WBUR's..."),
]


def draw(wav, tg, png, tmin, tmax, title):
    os.makedirs(os.path.dirname(png), exist_ok=True)
    wav, tg, png = map(os.path.abspath, (wav, tg, png))  # Praat resolves relative paths from the script folder
    title = title.replace("_", "\\_ ")  # a bare "_" is a subscript marker in Praat text
    res = subprocess.run([PRAAT, "--run", DRAW, wav, tg, png, "%.3f" % tmin, "%.3f" % tmax, title],
                         capture_output=True)
    if res.returncode != 0 or not os.path.exists(png):
        sys.exit("Praat failed on %s: %s" % (tg, res.stderr.decode("utf-8", "replace")[-500:]))


def wav_of(utt):
    return glob.glob(os.path.join(CORPUS, "*", utt + ".wav"))[0]


def tg_of(exp, utt):
    return glob.glob(os.path.join(TEXTGRIDS, exp, "*", utt + ".TextGrid"))[0]


def combined_textgrid(utt, exps, path):
    """One TextGrid holding the reference word tier and the phone tier of each experiment."""
    first = textgrid.openTextgrid(tg_of(exps[0], utt), includeEmptyIntervals=True)
    out = textgrid.Textgrid()
    out.addTier(first.getTier("words").new(name="words"))
    for exp in exps:
        tg = textgrid.openTextgrid(tg_of(exp, utt), includeEmptyIntervals=True)
        out.addTier(tg.getTier("phones").new(name=exp))
    out.save(path, format="long_textgrid", includeBlankSpaces=True)


def main():
    exps = [e for e in ORDER if os.path.isdir(os.path.join(TEXTGRIDS, e))]
    for tg in sorted(glob.glob(os.path.join(TEXTGRIDS, MAIN, "*", "*.TextGrid"))):
        utt = os.path.splitext(os.path.basename(tg))[0]
        draw(wav_of(utt), tg, os.path.join(FIGURES, MAIN, utt + ".png"), 0, 0, "%s - %s" % (utt, MAIN))
    for name, utt, a, b, title in ZOOMS:
        for exp in ("arpa_baseline", "arpa_g2p"):
            draw(wav_of(utt), tg_of(exp, utt), os.path.join(FIGURES, "%s_%s.png" % (name, exp)), a, b,
                 "%s (%s): %s" % (utt, exp, title))
    tmp = os.path.join(WORK, "figures")
    os.makedirs(tmp, exist_ok=True)
    for name, utt, a, b, title in COMPARE:
        path = os.path.join(tmp, name + ".TextGrid")
        combined_textgrid(utt, exps, path)
        draw(wav_of(utt), path, os.path.join(FIGURES, "compare_%s.png" % name), a, b,
             "%s: phone tiers of each model" % title)
    print("figures in", FIGURES)


if __name__ == "__main__":
    main()
