# Multimodal emotion recognition on TESS

Speech-only, text-only and speech + text (fusion) emotion classifiers for the Toronto Emotional Speech
Set (TESS). Each pipeline has the blocks preprocessing -> feature extraction -> temporal / contextual
modelling -> (fusion) -> classifier. The report is in [`report/Report.pdf`](report/Report.pdf).

## Main results

Test set: 30 words that are never used in training (420 recordings, both speakers, 7 emotions).
Mean ± standard deviation over 5 training seeds. Chance level: 14.3%.

| Pipeline | Variant | Test accuracy | Macro F1 |
|---|---|---|---|
| Speech | WavLM layer 9 + BiLSTM + attention | 99.8% ± 0.2 | 0.998 |
| Speech | log-mel + BiLSTM + attention (baseline) | 99.7% ± 0.2 | 0.997 |
| Text | BERT + Transformer encoder | 14.3% ± 0.0 | 0.036 |
| Fusion | concatenation | 99.7% ± 0.2 | 0.997 |
| Fusion | gated | 99.7% ± 0.3 | 0.997 |

Every TESS sentence is "Say the word ___", spoken in all 7 emotions, so the text holds no emotion
information and the text model stays at chance. This is expected and is analysed in the report.

Extra test, trained on one speaker and tested on the other (harder):

| Variant | Older -> younger | Younger -> older | Both directions |
|---|---|---|---|
| speech, WavLM | 57.4% ± 10.0 | 45.5% ± 5.8 | 51.5% |
| speech, log-mel | 41.3% ± 3.7 | 34.0% ± 4.9 | 37.7% |
| fusion, concatenation | 52.7% ± 8.5 | 38.0% ± 2.8 | 45.3% |
| fusion, gated | 56.6% ± 12.8 | 47.8% ± 8.0 | 52.2% |
| linear baseline (WavLM average + logistic regression) | 74.8% | 65.7% | 70.2% |

All tables and plots are in [`Results/`](Results/).

## Layout

```
preprocessing/prepare_data.py      find, clean and split the TESS files
preprocessing/extract_features.py  WavLM, log-mel and BERT features (+ WavLM layer choice)
src/networks.py                    temporal, contextual, fusion and classifier blocks
src/pipeline.py                    training and evaluation shared by all pipelines
src/results.py                     tables, confusion matrices, training curves, t-SNE
models/speech_pipeline/            train.py, test.py
models/text_pipeline/              train.py, test.py
models/fusion_pipeline/            train.py, test.py
experiments/cross_speaker.py       train on one speaker, test on the other
experiments/fusion_reliance.py     how much the fusion models use each modality
experiments/prosody_stats.py       pitch and duration per emotion (used in the analysis)
experiments/linear_baseline.py     WavLM average + logistic regression, same test sets
colab_train.ipynb                  main training run on a Colab GPU
colab_cross_speaker.ipynb          cross-speaker and fusion reliance run on a Colab GPU
Results/                           accuracy tables, predictions, plots
report/Report.pdf                  the report
```

## Setup

Tested with Python 3.14 (local, CPU) and the Colab Python (GPU).

```
python -m venv .venv
.venv\Scripts\activate            (Linux / macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Data: download TESS (for example the Kaggle copy "Toronto emotional speech set (TESS)") and unpack it
anywhere under `data/raw/`. The folder layout does not matter; files are found by their names
`<speaker>_<word>_<emotion>.wav`. Exact duplicate copies inside the archive are skipped. The data is
not included in this repository.

## Run

```
python preprocessing/prepare_data.py        # cleaned 16 kHz audio, data/manifest.csv, word-disjoint split
python preprocessing/extract_features.py    # features/ (downloads WavLM and BERT from Hugging Face)

python models/speech_pipeline/train.py
python models/text_pipeline/train.py
python models/fusion_pipeline/train.py

python models/speech_pipeline/test.py
python models/text_pipeline/test.py
python models/fusion_pipeline/test.py

python experiments/cross_speaker.py
python experiments/fusion_reliance.py
python experiments/prosody_stats.py
python experiments/linear_baseline.py
```

Trained checkpoints are not included, so run each `train.py` before its `test.py`. For a quick
single-seed run, add `--seeds 0` to both `train.py` and `test.py`. Training uses a GPU when one is available.

Feature extraction ran on a CPU; training of all variants and seeds ran on a Colab T4 GPU (the two
notebooks show the exact steps). For Colab:

1. Run the two preprocessing scripts locally.
2. Zip this folder together with `features/` and `data/manifest.csv` as `colab_package.zip` and
   upload it to `MyDrive/emotion_2b/`.
3. Run `colab_train.ipynb`. It writes `colab_results.zip` to the same Drive folder.
4. Zip the `src/`, `models/` and `experiments/` folders as `colab_code_v2.zip`, upload it, and run
   `colab_cross_speaker.ipynb`.

The split and seeds are fixed, so the same commands give the same split and very close numbers (GPU
kernels are not bit-exact).

## Notes

- Split: words, not files, are divided into train / validation / test (140 / 30 / 30 words, seed 0).
  Every word is spoken by both speakers in all 7 emotions, so a file-level split would put the same
  word in training and testing.
- All model choices (early stopping, WavLM layer) use only the validation words, never the test words.
- The text input is the fixed TESS sentence. No emotion words are added to the text.
