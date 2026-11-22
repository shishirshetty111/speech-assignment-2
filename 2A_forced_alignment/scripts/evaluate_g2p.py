"""Evaluate our own G2P model on dictionary words it never saw, next to the pretrained G2P.

Test words: 2,000 random english_us_arpa words that are NOT in the 50,000-word training sample
(see train_g2p.py). For each word the predicted pronunciation is compared with the dictionary:
  word error rate (WER)   share of words whose prediction matches none of the dictionary pronunciations
  phone error rate (PER)  phone edit distance to the closest dictionary pronunciation / its length
The pretrained english_us_arpa G2P was most likely trained on the whole dictionary, test words
included, so its numbers are optimistic and only a rough reference.

MFA's own `train_g2p --evaluate` reported 100% WER and 100% PER although the model predicts
correct pronunciations (e.g. hello -> HH EH1 L OW0), so that step is not used.

    python scripts/evaluate_g2p.py            # --reuse: score saved predictions again
"""
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dictionaries import base_dict_path, read_dict  # noqa: E402
from common import ANALYSIS, WORK, run_mfa  # noqa: E402

N_TEST = 2000
MODELS = {"own G2P (50,000 words)": os.path.join(WORK, "models", "g2p_own_arpa.zip"),
          "pretrained english_us_arpa G2P": "english_us_arpa"}


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main():
    full = read_dict(base_dict_path("english_us_arpa"))
    train = glob.glob(os.path.join(WORK, "dicts", "g2p_train_*.dict"))
    if not train:  # rebuild the same fixed training sample (seed 0) when training was skipped
        from train_g2p import subset_dictionary
        train = [subset_dictionary(50000)[0]]
    seen = set(read_dict(train[0]))
    # plain alphabetic words only, so the test is about spelling, not punctuation
    unseen = sorted(w for w in full if w not in seen and w.isalpha())
    test = random.Random(1).sample(unseen, N_TEST)
    tmp = os.path.join(WORK, "g2p_eval")
    os.makedirs(tmp, exist_ok=True)
    words_in = os.path.join(tmp, "test_words.txt")
    with open(words_in, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(test) + "\n")

    rows = []
    for name, model in MODELS.items():
        out = os.path.join(tmp, name.split()[0] + ".dict")
        if "--reuse" not in sys.argv or not os.path.exists(out):
            run_mfa("g2p", words_in, model, out, "--clean", "--num_pronunciations", "1",
                    "--temporary_directory", os.path.join(tmp, "tmp_" + name.split()[0]))
        pred = read_dict(out)
        row = [name]
        for strip in (False, True):  # with and without the stress digits (AH0/AH1 -> AH)
            norm = (lambda ph: [x.rstrip("012") for x in ph]) if strip else (lambda ph: ph)
            wrong, errs, length = 0, 0, 0
            for w in test:
                if w not in pred:
                    wrong += 1
                    continue
                p = norm(pred[w][0][1].split())
                best = min((edit_distance(p, norm(ref.split())), len(ref.split())) for _, ref in full[w])
                wrong += best[0] > 0
                errs += best[0]
                length += best[1]
            row += [100 * wrong / len(test), 100 * errs / max(length, 1)]
        rows.append(tuple(row) + (sum(w not in pred for w in test),))
    path = os.path.join(ANALYSIS, "g2p_evaluation.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("Test: %d english_us_arpa words not in the own model's training sample.\n\n" % N_TEST)
        f.write("| G2P model | WER | PER | WER ignoring stress | PER ignoring stress | no output |\n"
                "|---|---|---|---|---|---|\n")
        f.write("".join("| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %d |\n" % r for r in rows))
    print(open(path, encoding="utf-8").read())


if __name__ == "__main__":
    main()
