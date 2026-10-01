"""Create a tiny FAKE dataset with the same file layout as the Zindi challenge.

Used only to check that every notebook runs end-to-end (e.g. in CI, or before you have
downloaded the real data). Results on this data mean nothing.

    python tests/make_fake_dataset.py --out /tmp/fake_data --per-class 25
    DATA_DIR=/tmp/fake_data QUICK=1 python -m pytest   (or run the notebooks)

Each fake 'word' is a sequence of 2-3 tones, and some words contain the same tones in a
different ORDER - so an order-free baseline should do worse than the sequential models.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

WORDS = ["ndio", "hapana", "moja", "mbili", "tatu", "nne", "tano", "sita", "saba", "nane", "tisa", "kumi"]
PATTERNS = [(300, 600), (600, 300), (400, 800), (800, 400), (300, 500, 700), (700, 500, 300),
            (500, 900), (900, 500), (350, 1000), (1000, 350), (450, 650, 450), (650, 450, 650)]


def make_clip(pattern, sr, rng):
    parts = []
    for f in pattern:
        dur = rng.uniform(0.12, 0.25)
        t = np.arange(int(dur * sr)) / sr
        f = f * rng.uniform(0.9, 1.1)                          # 'speaker' pitch variation
        parts.append(np.sin(2 * np.pi * f * t) * np.hanning(len(t)))
    word = np.concatenate(parts)
    lead, trail = rng.uniform(0.05, 0.6, size=2)
    y = np.concatenate([np.zeros(int(lead * sr)), word, np.zeros(int(trail * sr))])
    y += rng.normal(0, rng.uniform(0.002, 0.05), size=len(y))  # recording noise
    return (0.5 * y / np.abs(y).max()).astype(np.float32)


def main(out, per_class, n_test, seed=0):
    rng = np.random.default_rng(seed)
    audio = Path(out) / "Audio"
    audio.mkdir(parents=True, exist_ok=True)
    rows, test = [], []
    for w, p in zip(WORDS, PATTERNS):
        for _ in range(per_class):
            name = f"{rng.integers(1e9):09d}.wav"
            sr = int(rng.choice([16000, 22050, 44100]))
            sf.write(audio / name, make_clip(p, sr, rng), sr)
            rows.append({"AUDIO": name, "Word": w})
    for _ in range(n_test):
        i = rng.integers(len(WORDS))
        name = f"t{rng.integers(1e9):09d}.wav"
        sf.write(audio / name, make_clip(PATTERNS[i], 16000, rng), 16000)
        test.append({"AUDIO": name})
    pd.DataFrame(rows).sample(frac=1, random_state=seed).to_csv(Path(out) / "Train.csv", index=False)
    pd.DataFrame(test).to_csv(Path(out) / "Test.csv", index=False)
    sub = pd.DataFrame(test)
    for w in WORDS:
        sub[w] = 0
    sub.to_csv(Path(out) / "SampleSubmission.csv", index=False)
    print(f"Fake dataset with {len(rows)} train / {len(test)} test clips written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/fake_data")
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--n-test", type=int, default=40)
    a = ap.parse_args()
    main(a.out, a.per_class, a.n_test)
