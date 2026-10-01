"""Dynamic Time Warping nearest-template classifier (Sakoe & Chiba, 1978).

A classical, non-learned SEQUENTIAL baseline: it compares whole MFCC sequences while
allowing non-linear stretching in time, which directly addresses speaking-rate variation.
It learns nothing, so it shows how much of the problem is solved by alignment alone.
"""
import librosa
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def _normalise(seq):
    return (seq - seq.mean(1, keepdims=True)) / (seq.std(1, keepdims=True) + 1e-6)   # per-utterance CMVN


def dtw_cost(a, b, band=0.3):
    """Length-normalised DTW cost between two (F, T) sequences using cosine distance."""
    D, wp = librosa.sequence.dtw(X=a, Y=b, metric="cosine", global_constraints=True, band_rad=band)
    return D[-1, -1] / len(wp)


def select_templates(seqs, y, per_class, seed=42):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        cand = np.where(y == c)[0]
        idx.extend(rng.choice(cand, size=min(per_class, len(cand)), replace=False))
    return np.array(idx)


def dtw_class_scores(train_seqs, y_train, query_seqs, n_classes, per_class=20, k=5,
                     band=0.3, n_jobs=-1, seed=42):
    """For each query: mean DTW cost to its k nearest templates of each class, using a
    random subset of `per_class` training templates per class. Returns (N, n_classes)."""
    t_idx = select_templates(train_seqs, y_train, per_class, seed)
    templates = [_normalise(train_seqs[i]) for i in t_idx]
    t_labels = y_train[t_idx]

    def row(q):
        q = _normalise(q)
        costs = np.array([dtw_cost(q, t, band) for t in templates])
        scores = np.full(n_classes, np.inf)
        for c in range(n_classes):
            cc = np.sort(costs[t_labels == c])[:k]
            if len(cc):
                scores[c] = cc.mean()
        return scores

    S = np.array(Parallel(n_jobs=n_jobs)(delayed(row)(q) for q in tqdm(query_seqs, desc="DTW")))
    return np.where(np.isfinite(S), S, np.nanmax(S[np.isfinite(S)]) + 1)


def scores_to_proba(S, temperature):
    """Softmax over negative costs. The temperature is tuned on the validation set, because
    the official metric (log loss) needs calibrated probabilities, not just a ranking."""
    logits = -S / temperature
    logits -= logits.max(1, keepdims=True)
    P = np.exp(logits)
    return P / P.sum(1, keepdims=True)
