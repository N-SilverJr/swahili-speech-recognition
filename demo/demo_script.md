# Demo video script (target 9 minutes, hard limit 7–10)

Each member speaks for about 2–2.5 minutes. Put these notes in your own words, since the rubric grades
*conceptual understanding*, and reading a script word for word sounds like it. Show the notebook or
figure named in each bullet while you talk. Replace every **[X]** with your real result.

Recording tips: one shared screen (Google Meet or Zoom recording works), cameras on during your own
segment, 1080p, and rehearse once end-to-end with a timer.

---

## Member 1 · 0:00 – 2:30 · Problem, data and baselines

**Show:** title slide → Zindi page → EDA figures → baseline results.

- **Hook (20 s).** "Many people in East Africa use services through a basic phone. A voice menu that
  understands *ndio*, *hapana* and the numbers in Swahili makes those services usable without reading
  or typing English. Swahili has over 100 million speakers, yet little labelled speech data."
- **Task and research question (20 s).** There are 12 Swahili words from about 300 Kenyan speakers
  recorded on their own devices. The metric is log loss. Our question is how well different
  *sequential* models handle this, and why.
- **Why it is a sequence problem (20 s).** Show a spectrogram: a word is an ordered series of sounds;
  speakers talk at different speeds and start at different times.
- **EDA findings, each tied to a decision (60 s):**
  - Duration and silence plot: **[X]%** silence, so we trim it. The 99th percentile gives a
    **[X]** s input length, which is **[X]** frames.
  - Mixed sample rates, so everything is resampled to 16 kHz, which is also what wav2vec expects.
  - **[X]%** noisy or clipped clips are kept on purpose, because real users sound like that.
  - No speaker IDs, so our test scores are on seen speakers. This is a limitation we come back to.
  - The similarity heatmap predicts which words will be confused.
- **Baselines (30 s).** The order-free SVM reached log loss **[X]** and DTW alignment reached **[X]**.
  "This is how far you get *without* learning the sequence, or *without learning at all*." Hand over to
  Member 2.

## Member 2 · 2:30 – 4:45 · Literature and the BiRNN with attention

**Show:** related-work diagram or slide → notebook 04 → attention figure.

- **Literature in one minute.** Isolated-word recognition started with templates and DTW (Sakoe &
  Chiba), then moved to CNNs for keyword spotting (Sainath & Parada), CNN plus RNN with attention (de
  Andrade et al.), and self-attention and self-supervised pretraining (Keyword Transformer; wav2vec 2.0;
  XLS-R). African-language work (Doumbouya et al.) shows pretraining helps where labels are scarce.
  "This gave us three truly different ways to read a sequence: local, recurrent and global."
- **Why a BiRNN (30 s).** It reads frame by frame with memory. The gates keep information across the
  ~**[X]** frames. Bidirectional means the end of the word informs the start. Attention learns which
  frames matter and ignores silence.
- **Experiments (45 s).** Walk through the log: mean pooling → attention (**[X]** change) → LSTM vs GRU →
  unidirectional → augmentation. Name **one result that surprised you** and explain why you think it
  happened.
- **Attention figure (15 s).** "Here the model focuses on the voiced part; here, in an error, it
  focused on noise."

## Member 3 · 4:45 – 6:45 · CNN, metrics and the comparison

**Show:** notebook 03 learning curves → comparison table → confusion matrices.

- **Why a CNN (30 s).** Words are made of local time-frequency events. 3×3 filters with pooling find them
  wherever they occur, which handles shifted onsets.
- **Progression (30 s).** CNN-01 overfitting (show the learning curve gap) → SpecAugment → label
  smoothing for log loss. State what each change did, with numbers.
- **Metrics (30 s).** Log loss is the official metric and rewards knowing when you are unsure, which
  matters for a voice menu that can ask the caller to repeat. Macro-F1 checks every word, and McNemar
  checks whether differences are real.
- **Comparison table (30 s).** Give the ranking and the size of the gaps. "The biggest jump is between
  **[X]** and **[X]**, which tells us **[X]**."

## Member 4 · 6:45 – 9:15 · Transformer, errors, conclusions

**Show:** notebook 05 → error-analysis figures → hardest clips (play one!) → conclusion slide.

- **Why wav2vec 2.0 / XLS-R (30 s).** It is pretrained on huge amounts of unlabelled speech, and XLS-R
  includes Swahili. Self-attention relates every frame to every other. Frozen vs fine-tuned: **[X]**.
  English vs multilingual: **[X]**. The cost is **[X]M** parameters vs **[X]K** for the CNN.
- **Error analysis (60 s).**
  - The most confused pair is **[X]/[X]**. Explain the reason acoustically.
  - Accuracy drops in the lowest-SNR quintile, from **[X]** to **[X]**. The transformer is more or less
    robust.
  - **Play one hard clip** and categorise it: noisy, different word or mislabelled.
- **Limitations (20 s).** Seen speakers, a single split, a closed vocabulary, and Kenyan speakers only.
- **Conclusion (30 s).** Answer the research question in two sentences, then give two future-work items
  (speaker-independent evaluation; distilling the transformer into a small on-device model).
- **Close (5 s).** "Thank you. The code, report and experiment logs are in our repository."

---

## Checklist before recording

- [ ] Every number above replaced with a real result.
- [ ] Each member can answer "why did you choose this model?" and "what would you change?" without notes.
- [ ] Total time 7–10 minutes.
- [ ] Video link added to the report and README.
