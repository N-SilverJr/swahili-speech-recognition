# Sequential Models for Swahili Spoken-Word Classification

Formative Assignment 2 · *Research-Informed Sequential Models for NLP and Language Technologies*
Dataset: [Zindi – Swahili Audio Classification](https://zindi.africa/competitions/swahili-audio-classification)
(12 Swahili words recorded by ~300 speakers in Kenya; official metric: multi-class log loss)

**Research question.** *How effectively can sequential modelling approaches classify isolated spoken
Swahili words recorded on everyday devices, and what evidence explains the strengths and limitations of
each approach?*

| Links | |
|---|---|
| Report (PDF) | `report/report.pdf` |
| Demo video (7–10 min) | _add link_ |
| Contribution tracker | _add link_ |

---

## The five approaches

| # | Approach | Input sequence | What it models | Notebook | Owner |
|---|---|---|---|---|---|
| B1 | MFCC summary statistics + RBF-SVM | stats over time (order discarded) | nothing sequential; a reference point | `02_baselines` | Member 1 |
| B2 | DTW nearest-template (k-NN) | MFCC+Δ+ΔΔ frames | sequence **alignment** without learning | `02_baselines` | Member 1 |
| N1 | 2-D CNN | log-mel spectrogram | **local** time-frequency patterns | `03_cnn_logmel` | Member 3 |
| N2 | BiGRU/BiLSTM + attention | MFCC+Δ+ΔΔ frames | **ordered** dependencies in both directions, learned frame weighting | `04_birnn_attention` | Member 2 |
| N3 | wav2vec 2.0 / XLS-R (fine-tuned) | raw waveform | **global** self-attention over self-supervised pretrained representations | `05_wav2vec2_transformer` | Member 4 |

## Results (held-out test split)

_Filled automatically by `06_compare_and_error_analysis.ipynb` → `results/comparison/comparison_table.md`.
Paste the table here after the final run._

## Repository structure

```
├── notebooks/
│   ├── 01_eda.ipynb                        # EDA + creates the shared split (Member 1)
│   ├── 02_baselines.ipynb                  # SVM and DTW baselines (Member 1)
│   ├── 03_cnn_logmel.ipynb                 # CNN experiments (Member 3)
│   ├── 04_birnn_attention.ipynb            # BiRNN + attention experiments (Member 2)
│   ├── 05_wav2vec2_transformer.ipynb       # wav2vec 2.0 / XLS-R experiments (Member 4)
│   └── 06_compare_and_error_analysis.ipynb # comparison, significance tests, error analysis
├── src/
│   ├── config.py      # ALL shared settings (sample rate, frame sizes, clip length, seed, paths)
│   ├── data.py        # CSV parsing, audio path lookup, shared stratified split
│   ├── features.py    # silence trimming, log-mel, MFCC, summary stats, caching, audio statistics
│   ├── datasets.py    # PyTorch datasets, SpecAugment, waveform augmentation
│   ├── models.py      # CNN, BiRNN+attention, Hugging Face transformer wrapper
│   ├── train.py       # one training loop for all neural models (AdamW, one-cycle, early stopping)
│   ├── pipeline.py    # experiment runner + final test evaluation
│   ├── dtw.py         # DTW template-matching classifier
│   ├── evaluate.py    # metrics, plots, McNemar test, experiment logging, Zindi submission
│   └── utils.py       # seeding, device, Drive backup
├── data/splits.csv            # shared split (committed); audio itself is NOT committed
├── experiments/               # one experiment log per model + combined experiment_log.csv
├── results/                   # metrics.json, predictions, figures per model; eda/; comparison/
├── submissions/               # Zindi submission files
├── report/                    # report draft and final PDF
├── demo/demo_script.md        # speaking script for the video
├── tests/make_fake_dataset.py # synthetic dataset for smoke-testing the pipeline
└── CONTRIBUTIONS.md
```

## How to run (Google Colab)

1. **Get the data.** Join the challenge on Zindi and download `Train.csv`, `Test.csv`,
   `SampleSubmission.csv` and `Audio.zip`. Upload all four to a Google Drive folder named
   **`swahili_audio`** (each member does this in their own Drive).
2. **Edit `REPO_URL`** in the first code cell of each notebook (once, then push).
3. **Run in this order.** Open each notebook in Colab (*File → Open notebook → GitHub*), select a
   **T4 GPU** for notebooks 03–05, and *Run all*:
   1. `01_eda` (Member 1) creates `data/splits.csv`. **Commit it before anyone trains.**
   2. `02`, `03`, `04`, `05` can then run **in parallel**, one per member.
   3. `06_compare_and_error_analysis` runs last, after everyone has pushed their `results/` and `experiments/` folders.
4. **Save outputs.** Colab deletes files when the session ends. The last cell of every notebook copies
   `results/`, `experiments/` and `submissions/` to `MyDrive/swahili_audio/outputs/`. Upload those
   folders to GitHub (drag-and-drop in the web UI works).

Approximate runtimes on a Colab T4: EDA ~10 min · baselines ~15 min (DTW is CPU-bound) · CNN ~15 min ·
BiRNN ~20 min · wav2vec 2.0 base ~30 min · XLS-R-300M ~1.5 h (all experiments).

### Local run / smoke test (no real data needed)

```bash
pip install -r requirements.txt torch
python tests/make_fake_dataset.py --out /tmp/fake_data
DATA_DIR=/tmp/fake_data QUICK=1 TINY_W2V=1 jupyter nbconvert --execute --to notebook notebooks/0*.ipynb
```
`QUICK=1` trains for 2 epochs and `TINY_W2V=1` uses a tiny random transformer. **Never report results
from these modes.**

## Reproducibility

* One configuration file (`src/config.py`) and one split file (`data/splits.csv`) for all models.
* Fixed seed (42) for Python, NumPy and PyTorch; deterministic cuDNN.
* Normalisation statistics are computed on the training split only.
* Model selection uses validation **log loss** only; the test split is evaluated **once** per model.
* Every run is logged with its hypothesis, motivation, configuration, metrics, time and parameter count
  in `experiments/log_<model>.csv`.
* Features are cached, keyed by a hash of all settings, so a config change never silently reuses stale features.

## Citation of the data

Zindi. (2022). *Swahili Audio Classification* [Data set].
https://zindi.africa/competitions/swahili-audio-classification
