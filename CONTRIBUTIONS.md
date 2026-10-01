# Individual contributions

Copy these rows into the official Google Sheets tracker. Replace "Member N" with names and keep
the entries factual: what you built, which experiments you ran (IDs from `experiments/`), and which
report/demo sections you wrote.

| Member | Model(s) trained | Experiments (IDs) | Code / notebooks | Report sections | Demo segment |
|---|---|---|---|---|---|
| Member 1 | MFCC-stats + SVM; DTW k-NN | BL-01 – BL-03, DTW-01 – DTW-02 | `01_eda`, `02_baselines`, `src/data.py`, `src/features.py`, `src/dtw.py` | Introduction; Dataset & EDA | 0:00 – 2:30 Problem, data, baselines |
| Member 2 | BiGRU/BiLSTM + attention | RNN-01 – RNN-06 | `04_birnn_attention`, `src/models.py` (BiRNN), `src/datasets.py` | Related Work; Methodology (RNN) | 2:30 – 4:30 Literature + BiRNN |
| Member 3 | 2-D CNN on log-mel | CNN-01 – CNN-05 | `03_cnn_logmel`, `src/train.py`, `src/evaluate.py`, `06` (comparison part) | Evaluation metrics; Results & Discussion | 4:30 – 6:30 CNN, metrics, results |
| Member 4 | wav2vec 2.0 / XLS-R | W2V-01 – W2V-04 | `05_wav2vec2_transformer`, `src/pipeline.py`, `06` (error analysis), README | Error Analysis & Limitations; Conclusion | 6:30 – 9:30 Transformer, errors, conclusions |

Shared by everyone: reviewing each other's notebooks, writing the `observation` column for their own
experiments, proofreading the full report, and being able to explain every part of the project in the demo.
