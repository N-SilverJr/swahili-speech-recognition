---
title: "Sequential Models for Isolated Spoken-Word Classification in Swahili"
subtitle: "Formative Assignment 2 · Research-Informed Sequential Models for NLP and Language Technologies"
author: "Group [N]: [Member 1], [Member 2], [Member 3], [Member 4]"
date: "[Month Year]"
---

<!--
HOW TO USE THIS DRAFT
* Values in double curly braces are filled automatically by `python report/fill_report.py`
  from results/eda/eda_summary.json and results/comparison/comparison_table.csv.
* Paragraphs marked [[WRITE: ...]] must be written by the group from your own results.
  They cannot be written before the experiments are run, and they carry the most marks.
* Before submitting: rewrite every section in your own voice, delete these comments, and export to PDF.
-->

**Links.** Code: [GitHub repository URL] · Demo video: [URL] · Contribution tracker: [URL]

# 1 Introduction

Voice interfaces allow people to use digital services without reading or typing. That matters in East
Africa, where Swahili has more than 100 million speakers (Zindi, 2022), many people first reach the
internet through basic phones, and literacy in the language of most software (English) cannot be
assumed. The simplest useful voice interface recognises a small, fixed vocabulary: "yes", "no" and the
digits are enough to navigate an interactive voice-response menu, confirm a mobile-money transaction or
answer a health survey. Commercial speech recognisers support Swahili poorly compared with English, and
publicly available labelled Swahili speech is scarce, so building even this small-vocabulary capability
is a low-resource problem.

We study the Zindi *Swahili Audio Classification* challenge: each clip contains one of twelve Swahili
words recorded by one of roughly 300 speakers in Kenya on their own devices, and the task is to predict
which word was spoken. Although it is posed as classification, the input is a **sequence**: a word is an
ordered series of sounds that unfolds over roughly half a second to a second, is stretched or compressed
by the speaker's rate of speech, and is shifted in time by leading silence. Two words can share most of
their sounds and differ mainly in their order or duration. How a model represents and aggregates that
sequence is therefore the central design decision.

**Research question.** *How effectively can sequential modelling approaches classify isolated spoken
Swahili words recorded on everyday devices, and what evidence explains the strengths and limitations of
each approach?*

We compare five approaches: an order-free baseline, a classical alignment baseline and three neural
architectures that model sequential structure in different ways (local convolutions, bidirectional
recurrence with attention, and a pretrained self-attention transformer). All five use identical data
splits, preprocessing settings and evaluation code, so the differences we observe can be attributed to
the models. Our contributions are (i) an exploratory analysis of the recordings as sequences, (ii) a
controlled comparison with {{n_experiments}} logged experiments, and (iii) an error analysis linking
failures to properties of the audio.

# 2 Related Work

**Isolated-word and keyword recognition.** Before deep learning, isolated-word recognisers compared a
spoken input to stored templates with **dynamic time warping** (DTW), which aligns two sequences by
non-linearly stretching time to absorb differences in speaking rate (Sakoe & Chiba, 1978). The standard
frame representation is the mel-frequency cepstral coefficient (MFCC), which summarises the short-time
spectrum on a perceptual frequency scale (Davis & Mermelstein, 1980). DTW needs no training, but it
learns no invariance to speaker, channel or noise, which is why it degrades with many speakers. We keep
it as a baseline because it isolates the value of *alignment* from the value of *learning*.

**Neural keyword spotting.** Sainath and Parada (2015) showed that convolutional networks over
spectrograms outperform fully connected networks for small-footprint keyword spotting, because
convolutions share weights across time and frequency and tolerate small shifts. The Speech Commands
dataset (Warden, 2018), with tens of thousands of one-second utterances of short English words, became
the standard benchmark and is structurally the closest public analogue to our task. On it, de Andrade et
al. (2018) combined convolutions with bidirectional LSTMs and an attention layer. The attention weights
both improved accuracy and showed which part of the utterance the model relied on. Berg et al. (2021)
later showed that a pure self-attention model over spectrogram patches can match or beat convolutional
and recurrent models on the same benchmark. Together these works identify three distinct ways of
modelling the sequence (local convolution, recurrence and self-attention), which we adopt as our three
neural approaches.

**Recurrent models.** Gated recurrent networks (LSTM: Hochreiter & Schmidhuber, 1997; GRU: Cho et al.,
2014) were designed to carry information across long sequences without vanishing gradients. Chung et al.
(2014) found no consistent winner between the two gates, which motivates testing both rather than
assuming one. Attention pooling (Bahdanau et al., 2015) replaces the final hidden state with a learned
weighted average over time steps, which suits our clips because a variable share of every clip is
silence or padding.

**Self-supervised speech representations and low-resource languages.** wav2vec 2.0 (Baevski et al.,
2020) pretrains a convolutional encoder and a transformer on unlabelled audio by solving a contrastive
task over masked latent frames. Fine-tuned with very little labelled data, it achieved strong speech
recognition, which is exactly the regime of low-resource languages. XLS-R (Babu et al., 2022) extends
this to 128 languages and about 436,000 hours of speech from sources that include Swahili (e.g., Common
Voice: Ardila et al., 2020), and MMS (Pratap et al., 2024) scales speech technology to over a thousand
languages. For African languages specifically, Doumbouya et al. (2021) showed that self-supervised
pretraining on unlabelled West African radio archives improved spoken-command recognition in languages
with almost no labelled data. Olatunji et al. (2023) showed that general-purpose models trained mostly on
Western speech perform worse on African-accented speech. Swahili ASR resources have been built for over
a decade (Gelas et al., 2012), but labelled data remain small. On this very dataset, the winning solution
of the related Zindi hackathon used a pretrained wav2vec 2.0 encoder with a classification head (Zindi,
2023). These findings lead us to include a pretrained transformer and to test explicitly whether
**multilingual** pretraining (XLS-R) transfers better than **English-only** pretraining (wav2vec 2.0 Base).

**Data augmentation.** SpecAugment (Park et al., 2019) masks random frequency bands and time spans of
the spectrogram and is a standard regulariser for speech models. EDA (Section 3) shows large
within-class variation and shifted word onsets, so we test SpecAugment with random time shifts for the
spectral models and additive noise and gain for the waveform model.

**How the literature shaped our design.** (1) All three neural families are established for short-word
recognition, but they differ in *how* they aggregate the sequence, which gives a meaningful comparison.
(2) The low-resource setting and the prior Zindi result suggest the pretrained transformer should be
strongest; our interest is in *how much* stronger it is, at what cost, and on which clips. (3)
Order-free and alignment-only baselines let us test whether modelling order, and learning it, matter at
all.

# 3 Dataset and Exploratory Analysis

The labelled set contains {{n_labelled}} clips across {{n_classes}} words ({{classes}}). The
unlabelled Zindi test set contains {{n_zindi_test}} clips. Because the Zindi test labels are hidden, we
created a stratified 80/10/10 train/validation/test split of the labelled data
(train {{split_sizes.train}}, validation {{split_sizes.val}}, test {{split_sizes.test}}), fixed with a
seed and shared by all models. Validation is used for all model selection; the test split is evaluated
once per model.

**Class balance.** Class sizes range from {{class_count_min}} to {{class_count_max}} clips (ratio
{{imbalance_ratio_max_min}}; Figure 1). [[WRITE: one sentence on whether imbalance matters and what it
implies for metrics.]]

**Recording heterogeneity.** The clips arrive at several sample rates ({{orig_sample_rates}}). Every clip
is converted to mono 16 kHz. This rate preserves the speech band below 8 kHz and is the rate at which
wav2vec 2.0 and XLS-R were pretrained.

**Sequence length and silence.** The median raw clip lasts {{duration_raw.50%}} s, but after trimming
leading and trailing silence (30 dB below peak) the median spoken word lasts
{{duration_trimmed.50%}} s. The median clip is {{silence_ratio_median}} silence (Figure 2). The
99th percentile of spoken duration is {{duration_trimmed.99%}} s. We therefore fix inputs to
{{max_seconds}} s after trimming, which crops fewer than 1% of words. At a 10 ms hop this is
{{n_frames}} frames: long enough that a plain RNN would struggle to propagate information from the
first to the last frame (motivating gated units and attention pooling), yet short enough that a
four-block CNN's receptive field covers most of the word. Words also differ systematically in duration,
so duration is informative in its own right (included explicitly in the order-free baseline).
[[WRITE: which words are longest/shortest.]]

**Recording quality.** An energy-based SNR proxy has a median of {{snr_median}} dB. {{n_flagged}} clips
({{frac_flagged}}) are near-silent, very noisy or clipped (Figure 3). We deliberately **kept** them: the
deployed system will face the same conditions, and removing them would make evaluation optimistic.
Instead, we peak-normalise every clip, augment with noise and masking, and test in Section 6 whether
errors concentrate in these recordings. [[WRITE: what you heard when listening to flagged clips.]]

**Duplicates and leakage.** We found {{n_duplicate_files}} byte-identical files, and
{{n_duplicate_groups_across_splits}} duplicate groups spanning splits. The dataset provides **no speaker
identifiers**, so the same speaker can appear in training and test data. Our test scores therefore
estimate performance on *seen* speakers and are likely optimistic for new ones (Section 6).

**Acoustic similarity between words.** Figure 5 shows the cosine similarity of class-average,
order-free MFCC summaries. The most similar pairs are {{most_similar_pairs}}. If such pairs share sounds
but differ in their *order*, an order-free model should confuse them while sequential models should
not. We test this prediction in Section 5. A t-SNE projection of the same features (Figure 6) shows
[[WRITE: how much the classes overlap]].

*Figures: 1 `eda_class_distribution.png`, 2 `eda_durations.png`, 3 `eda_quality.png`,
4 `eda_spectrograms_per_word.png` + `eda_within_class_variability.png`, 5 `eda_class_similarity.png`,
6 `eda_tsne.png` (all in `results/figures/`).*

# 4 Methodology

## 4.1 Preprocessing and sequence representations

All approaches share one pipeline (`src/features.py`): load audio, then resample to 16 kHz mono, trim
silence, peak-normalise, and centre-crop or pad to {{max_seconds}} s. From this, three sequence
representations are derived with a 25 ms window and a 10 ms hop:

| Representation | Shape | Used by | Rationale |
|---|---|---|---|
| Log-mel spectrogram | 64 × {{n_frames}} | CNN | preserves the full spectral detail as a time-frequency image |
| MFCC + Δ + ΔΔ | 39 × {{n_frames}} | BiRNN, DTW | compact, decorrelated frame vectors with local dynamics (Davis & Mermelstein, 1980) |
| Raw waveform | {{n_samples}} samples | wav2vec 2.0 / XLS-R | the model learns its own front-end |
| MFCC summary statistics | 131-dim vector | SVM | order deliberately discarded |

Spectral features are standardised per feature row, using statistics from the training split only.

## 4.2 The five approaches

**B1: Order-free baseline (MFCC statistics + RBF-SVM).** This model collapses each clip to the mean,
standard deviation, minimum and maximum of 20 MFCCs, delta statistics, spectral descriptors and
duration, then classifies with an RBF-kernel SVM (C and γ tuned by 5-fold cross-validation on log loss;
Platt-scaled probabilities). It answers the question: *how much can be done without the sequence?*

**B2: Alignment baseline (DTW k-NN).** Each query MFCC sequence is aligned to a random set of training
templates per class using DTW with cosine distance and a Sakoe–Chiba band. The class score is the mean
cost of the *k* nearest templates of that class, converted to probabilities by a softmax whose
temperature is tuned on validation.

**N1: 2-D CNN on log-mel.** The CNN has four blocks of two 3×3 convolutions with batch norm, ReLU, 2×2
max pooling and dropout, with channel widths 32–64–128–256. The output is averaged over frequency, then
max- and mean-pooled over time, and passed to a linear classifier. It models *local* sequential
structure: onsets and formant transitions within its receptive field, with invariance to small shifts.

**N2: BiGRU/BiLSTM with attention on MFCC sequences.** The network has two bidirectional recurrent
layers (128 units per direction), additive attention pooling over the 2×128-dimensional states, and a
two-layer classifier. It models the *ordered* dependency between sounds across the whole utterance and
in both directions, and it learns which frames to weight.

**N3: Pretrained transformer (wav2vec 2.0 Base / XLS-R-300M).** The encoder is a pretrained speech model
from Hugging Face Transformers (Wolf et al., 2020) with a mean-pooled classification head. The
convolutional feature encoder is frozen and the transformer layers are fine-tuned. It models *global*
dependencies through self-attention, using representations learned from large unlabelled corpora.

These are not variants of one architecture: they differ in what "sequence" they read (image, frame
vectors, raw samples), in their mechanism (local weight sharing, recurrence, self-attention) and in
whether they bring prior knowledge from pretraining.

## 4.3 Training and experimental design

All neural models use one training loop (`src/train.py`): AdamW, a one-cycle learning-rate schedule,
gradient clipping, mixed precision on GPU, and early stopping on **validation log loss** with the best
epoch restored. The seed is fixed at 42. Each approach went through a sequence of experiments that change
one factor at a time. Each experiment is logged with its hypothesis, the observation that motivated it,
configuration, validation metrics, training time and parameter count (`experiments/`). Factors studied
include:

* **Preprocessing:** silence trimming on/off (BL-03, CNN-05).
* **Representation:** MFCC vs log-mel for the RNN (RNN-06); raw waveform with English vs multilingual
  pretraining (W2V-02 vs W2V-03).
* **Architecture components:** mean vs attention pooling (RNN-01/02), GRU vs LSTM (RNN-03),
  uni- vs bidirectional (RNN-04), CNN width (CNN-04), frozen vs fine-tuned encoder (W2V-01/02).
* **Regularisation and augmentation:** SpecAugment/time shift (CNN-02, RNN-05), label smoothing
  (CNN-03), waveform noise and gain augmentation (W2V-04).
* **Baseline design:** linear vs non-linear (BL-01/02), templates per class and *k* (DTW-01/02).

[[WRITE: 3–5 sentences on how the progression actually went, e.g. "CNN-01 reached 99% training accuracy
but X% validation, so CNN-02 added augmentation, which ..." Use the observation column.]]

## 4.4 Evaluation metrics

**Log loss (primary).** Log loss is the official challenge metric, and it is a strictly proper scoring
rule (Gneiting & Raftery, 2007): it is minimised only by reporting true probabilities, so it rewards
calibrated confidence, not just the correct top guess. This matters in deployment. A voice menu that
knows when it is unsure can ask the caller to repeat the word instead of acting on a wrong guess. Its
weakness is that a few confidently wrong predictions dominate it, which makes it sensitive to label
noise. We therefore also report:

* **Accuracy and macro-F1.** Accuracy is directly interpretable. Macro-F1 weights every word equally, so
  a model cannot hide poor performance on one word behind the others (Opitz & Burst, 2019). The two
  should agree when classes are balanced, and a gap between them signals uneven per-class performance.
* **Top-2 accuracy and one-vs-rest ROC-AUC.** These measure ranking quality, relevant for a system that
  offers the two most likely words for confirmation.
* **Expected calibration error (ECE) and reliability diagrams.** These explain *why* log loss and
  accuracy can rank models differently (Guo et al., 2017).
* **Confusion matrices and per-class F1.** These are the basis for the error analysis.
* **Exact McNemar tests** on paired test predictions (Dietterich, 1998). They tell us whether differences
  between models are larger than chance on a test set of {{split_sizes.test}} clips.

A limitation of this evaluation is that the test split contains the same speakers as training, which
inflates scores relative to deployment on new speakers.

# 5 Results and Discussion

**Table 1.** Test-split performance of the selected configuration of each approach (↓ lower is better).

{{comparison_table}}

*Figures: `results/comparison/comparison_bars.png`, `experiment_progression.png`, `per_class_f1.png`,
`confusions_all_models.png`, `reliability_all.png`, `params_vs_logloss.png`; per-model learning curves,
confusion matrices and ROC curves in `results/<model>/`.*

[[WRITE the discussion around these questions, using your numbers. Suggested paragraph order:]]

1. **Does modelling order help?** Compare B1 (order-free) with B2 and N1–N3. Do the SVM's worst
   confusions coincide with the most similar pairs from EDA, and do the sequential models resolve them?
   Support this with the per-class F1 heatmap and the McNemar p-values.
2. **Does learning help beyond alignment?** Compare B2 (DTW) with N1/N2. Relate the result to the ~300
   speakers: DTW has no speaker invariance.
3. **Local vs recurrent vs global.** Compare N1, N2 and N3. What did the ablations show: attention vs
   mean pooling, bidirectionality, GRU vs LSTM? Relate to the literature (e.g., Chung et al., 2014, on
   gating; Berg et al., 2021, on self-attention).
4. **The value of pretraining.** Compare frozen vs fine-tuned and English vs multilingual pretraining.
   Is the gain consistent with Baevski et al. (2020) and Babu et al. (2022)?
5. **Preprocessing effects.** What did trimming and augmentation do, and does it match the EDA
   prediction?
6. **Calibration and log loss.** Does any model have high accuracy but worse log loss? Use the
   reliability diagram and the label-smoothing experiment.
7. **Cost.** Parameters, training time and inference constraints (a phone or IVR server) against
   the accuracy gain.

# 6 Error Analysis and Limitations

[[WRITE using notebook 06 outputs in `results/comparison/`:]]

* **Systematic confusions** (`most_confused_pairs.csv`): which pairs, for which models, and why
  acoustically. Plot or describe the spectrograms of one confused pair.
* **Where errors concentrate** (`accuracy_by_snr_duration.png`, `hard_vs_easy_audio_stats.csv`): accuracy
  by SNR and by spoken-duration quintile. Is the pretrained model more robust to noise?
* **Universally hard clips** (`hardest_clips_to_categorise.csv`): listen to them and categorise each as
  noisy, truncated, a different word spoken, likely mislabelled, or ambiguous. Report the counts.
* **Confidently wrong predictions** (`confidently_wrong_best_model.csv`): how many are flagged in EDA or
  sound mislabelled? This quantifies label noise and explains part of the log loss.
* **Attention behaviour** (`results/birnn_attention/attention_examples.png`): in errors, does attention
  land on noise instead of the word?

**Limitations of the study.** (i) No speaker IDs means speaker-independent evaluation is impossible, so
scores are likely optimistic. (ii) A single split and seed means the variance across seeds is not
estimated; small differences between models should be read with the McNemar results in mind. (iii) The
twelve-word closed vocabulary means results do not transfer to open-vocabulary recognition or to
rejecting out-of-vocabulary speech. (iv) All speakers are from Kenya, so dialects and accents from
Tanzania, Uganda or the DRC are not represented. (v) Compute limits (a single Colab T4) restricted
hyperparameter search for the 300M-parameter model.

# 7 Conclusion and Future Work

[[WRITE 1 paragraph: answer the research question directly, with the key numbers from Table 1 and the
two or three findings you are most confident in.]]

Future work: (1) speaker-independent evaluation, by requesting or reconstructing speaker metadata;
(2) seed and cross-validation repeats to estimate variance; (3) knowledge distillation from the
transformer into the small CNN/RNN for on-device use; (4) ensembling the complementary models (our
exploratory mean-probability ensemble reached a test log loss of {{ensemble_log_loss}}, not used for
model selection); (5) an "unknown word / silence" class, required for real IVR use; (6) collecting
recordings from other Swahili-speaking regions.

# References

Ardila, R., Branson, M., Davis, K., Kohler, M., Meyer, J., Henretty, M., Morais, R., Saunders, L., Tyers,
F., & Weber, G. (2020). Common Voice: A massively-multilingual speech corpus. In *Proceedings of the 12th
Language Resources and Evaluation Conference* (pp. 4218–4222). European Language Resources Association.

Babu, A., Wang, C., Tjandra, A., Lakhotia, K., Xu, Q., Goyal, N., Singh, K., von Platen, P., Saraf, Y.,
Pino, J., Baevski, A., Conneau, A., & Auli, M. (2022). XLS-R: Self-supervised cross-lingual speech
representation learning at scale. In *Proceedings of Interspeech 2022* (pp. 2278–2282).

Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised
learning of speech representations. In *Advances in Neural Information Processing Systems 33*
(pp. 12449–12460).

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and
translate. In *International Conference on Learning Representations*.

Berg, A., O'Connor, M., & Cruz, M. T. (2021). Keyword Transformer: A self-attention model for keyword
spotting. In *Proceedings of Interspeech 2021* (pp. 4249–4253).

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).
Learning phrase representations using RNN encoder–decoder for statistical machine translation. In
*Proceedings of EMNLP 2014* (pp. 1724–1734).

Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). *Empirical evaluation of gated recurrent neural
networks on sequence modeling* (arXiv:1412.3555). arXiv.

Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word
recognition in continuously spoken sentences. *IEEE Transactions on Acoustics, Speech, and Signal
Processing, 28*(4), 357–366.

de Andrade, D. C., Leo, S., Viana, M. L. D. S., & Bernkopf, C. (2018). *A neural attention model for
speech command recognition* (arXiv:1808.08929). arXiv.

Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning
algorithms. *Neural Computation, 10*(7), 1895–1923.

Doumbouya, M., Einstein, L., & Piech, C. (2021). Using radio archives for low-resource speech
recognition: Towards an intelligent virtual assistant for illiterate users. In *Proceedings of the AAAI
Conference on Artificial Intelligence, 35*(17), 14757–14765.

Gelas, H., Besacier, L., & Pellegrino, F. (2012). Developments of Swahili resources for an automatic
speech recognition system. In *Proceedings of the Workshop on Spoken Language Technologies for
Under-Resourced Languages (SLTU)*.

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation.
*Journal of the American Statistical Association, 102*(477), 359–378.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In
*Proceedings of the 34th International Conference on Machine Learning* (pp. 1321–1330).

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.

McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015).
librosa: Audio and music signal analysis in Python. In *Proceedings of the 14th Python in Science
Conference* (pp. 18–25).

Olatunji, T., Afonja, T., Yadavalli, A., Emezue, C. C., Singh, S., Dossou, B. F. P., Osuchukwu, J.,
Osei, S., Tonja, A. L., Etori, N., & Mbataku, C. (2023). AfriSpeech-200: Pan-African accented speech
dataset for clinical and general domain ASR. *Transactions of the Association for Computational
Linguistics, 11*, 1669–1685.

Opitz, J., & Burst, S. (2019). *Macro F1 and macro F1* (arXiv:1911.03347). arXiv.

Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). SpecAugment: A
simple data augmentation method for automatic speech recognition. In *Proceedings of Interspeech 2019*
(pp. 2613–2617).

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., et al. (2019). PyTorch: An
imperative style, high-performance deep learning library. In *Advances in Neural Information Processing
Systems 32* (pp. 8024–8035).

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011).
Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Pratap, V., Tjandra, A., Shi, B., Tomasello, P., Babu, A., Kundu, S., et al. (2024). Scaling speech
technology to 1,000+ languages. *Journal of Machine Learning Research, 25*(97), 1–52.

Sainath, T. N., & Parada, C. (2015). Convolutional neural networks for small-footprint keyword spotting.
In *Proceedings of Interspeech 2015* (pp. 1478–1482).

Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition.
*IEEE Transactions on Acoustics, Speech, and Signal Processing, 26*(1), 43–49.

Warden, P. (2018). *Speech Commands: A dataset for limited-vocabulary speech recognition*
(arXiv:1804.03209). arXiv.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., et al. (2020). Transformers:
State-of-the-art natural language processing. In *Proceedings of EMNLP 2020: System Demonstrations*
(pp. 38–45).

Zindi. (2022). *Swahili Audio Classification* [Data set and competition].
https://zindi.africa/competitions/swahili-audio-classification

Zindi. (2023, January 6). *Meet the winners of the Swahili Audio Classification Hackathon*.
https://zindi.africa/blog/meet-the-winners-of-the-swahili-audio-classification-hackathon
