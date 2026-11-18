# Forced Alignment with the Montreal Forced Aligner (MFA)

Word and phone alignment of the 6 provided recordings with MFA 3.4.3:

- 3 news paragraphs read by one professional speaker (BU Radio News, files `F2BJRLP1-3`, 25-31 s each);
- 3 short sentences read by a non-native English learner (ISLE, files `ISLE_SESS0131_BLOCKD02_01-03`, about 4 s each).

The main output is one TextGrid per file in `outputs/textgrids/arpa_g2p/`. The short report is `report/Report.pdf`.

## What the pipeline does

| Step | Script | What happens |
|---|---|---|
| 1 | `scripts/prepare_corpus.py` | Puts the data in MFA's layout (`data/corpus/<speaker>/<file>.wav + .lab`). The BU files are NIST SPHERE audio with a `.wav` name and are converted to standard WAV (samples unchanged). Transcript line breaks and split hyphens (`seventy- six`) are cleaned in the `.lab` copy; every change is logged in `outputs/analysis/text_changes.tsv`. No spelling is corrected. |
| 2 | `run_pipeline.py` | Downloads the MFA pretrained models: acoustic models, dictionaries and G2P models for `english_us_arpa` and `english_mfa`. |
| 3 | `scripts/train_g2p.py` | Trains our own G2P (spelling to pronunciation) model on a fixed random 50,000-word sample of the `english_us_arpa` dictionary, with 10% of those words held out for testing. `--words 0` uses the whole dictionary (needs much more memory and time). |
| 3b | `scripts/evaluate_g2p.py` | Tests the own G2P model and the pretrained one on 2,000 dictionary words the own model never saw (`outputs/analysis/g2p_evaluation.md`). |
| 4 | `scripts/build_dictionaries.py` | Finds the words missing from the dictionary with `mfa validate` and adds pronunciations: acronyms (WBUR) are spelled out letter by letter, other words come from a G2P model. |
| 5 | `scripts/align_all.py` | Aligns the corpus with each dictionary and acoustic model combination (below). |
| 6 | `scripts/analyse.py` | Statistics, boundary comparison between models, and vowel measurements with Praat (`outputs/analysis/`). |
| 7 | `scripts/make_figures.py` | Praat pictures: waveform, spectrogram, word and phone tiers (`outputs/figures/`). |

### Experiments

| Name | Dictionary | Acoustic model |
|---|---|---|
| `arpa_baseline` | `english_us_arpa`, missing words left as they are | `english_us_arpa` (pretrained) |
| `arpa_g2p` (main) | `english_us_arpa` + pretrained G2P for missing words | `english_us_arpa` (pretrained) |
| `arpa_own_g2p` | `english_us_arpa` + our own G2P for missing words | `english_us_arpa` (pretrained) |
| `mfa_g2p` | `english_mfa` + pretrained G2P for missing words | `english_mfa` (pretrained, IPA-style phones) |
| `arpa_adapted` | as `arpa_g2p` | `english_us_arpa` adapted to this data (`mfa adapt`) |
| `scratch` | as `arpa_g2p` | trained from scratch on these 97 s only (`mfa train`) |

## Setup

1. **Install MFA** (Windows, Linux or macOS) with conda or mamba from conda-forge:

   ```bash
   conda env create -f environment.yml
   conda activate mfa
   mfa version            # 3.4.3
   ```

   Or without the file: `conda create -n mfa -c conda-forge montreal-forced-aligner=3.4.3`.

2. **Install Praat** from https://www.praat.org. It is used to draw the figures and measure vowels, and to inspect the TextGrids by hand. If `praat` is not on the PATH, set `PRAAT` to the executable, for example `set PRAAT=C:\tools\praat\Praat.exe`.

3. **Add the data.** The audio and transcripts come from licensed corpora (BU Radio News Corpus, ISLE) and are not included. Copy the provided folders into `data/raw/`:

   ```
   data/raw/wav/F2BJRLP1.wav ...
   data/raw/transcripts/F2BJRLP1.TXT ...
   ```

## Run

Everything with one command (in the `mfa` environment):

```bash
python run_pipeline.py                      # about 1 h on a laptop CPU, most of it G2P training
python run_pipeline.py --skip_g2p_training  # about 30 min, leaves out the own-G2P experiment
```

Or step by step:

```bash
python scripts/prepare_corpus.py
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
python scripts/build_dictionaries.py --variant arpa_pretrained_g2p
python scripts/align_all.py --only arpa_g2p
```

The core MFA commands behind the scripts are:

```bash
mfa validate data/corpus english_us_arpa --ignore_acoustics             # lists missing words
mfa g2p oovs.txt english_us_arpa oov_pronunciations.dict                # pronunciations for them
mfa align data/corpus work/dicts/arpa_pretrained_g2p.dict english_us_arpa outputs/textgrids/arpa_g2p
```

## Outputs

```
outputs/
  textgrids/<experiment>/<speaker>/<file>.TextGrid   word and phone tiers (open in Praat with the .wav)
  textgrids/<experiment>/alignment_analysis.csv      MFA's own per-file quality figures
  analysis/summary.md                                tables used in the report
  figures/                                           Praat pictures
  models/g2p_own_arpa.zip                            our own trained G2P model
dictionaries/<variant>_additions.dict                pronunciations added for missing words
report/Report.pdf
```

To inspect a file in Praat: Open > Read from file, select the `.wav` and the `.TextGrid`, select both, then View & Edit.

## Licences

MFA is MIT licensed. The licences of the pretrained models are listed on their pages at https://mfa-models.readthedocs.io. The audio data is not redistributed here.
