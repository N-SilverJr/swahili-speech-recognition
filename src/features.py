"""Audio preprocessing and feature extraction.

Sequence representations used in the project
    raw waveform  (T_samples,)      -> wav2vec 2.0 / XLS-R
    log-mel       (N_MELS, T_frames) -> 2-D CNN
    MFCC+Δ+ΔΔ     (39, T_frames)     -> BiGRU/BiLSTM + attention, DTW
    MFCC stats    (D,)               -> SVM baseline (order discarded on purpose)
All features are cached to disk so members only pay the extraction cost once.
"""
import hashlib
import json
from pathlib import Path

import joblib
import librosa
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from . import config as C


# ------------------------------------------------------------------ waveform
def load_audio(path, sr=C.SR, trim=True, top_db=C.TOP_DB, normalize=True):
    """Load mono audio at `sr`, trim leading/trailing silence, peak-normalise."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return np.zeros(int(0.1 * sr), dtype=np.float32)
    if trim:
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        if len(yt) >= int(0.1 * sr):          # never trim a clip down to almost nothing
            y = yt
    if normalize:
        peak = np.max(np.abs(y))
        if peak > 0:
            y = 0.95 * y / peak
    return y.astype(np.float32)


def fix_length(y, n):
    """Centre-crop or centre-pad a waveform (or frame axis of a feature) to n."""
    if len(y) >= n:
        s = (len(y) - n) // 2
        return y[s:s + n]
    pad = n - len(y)
    return np.pad(y, (pad // 2, pad - pad // 2))


def n_samples(seconds=None):
    return int(round((seconds or C.MAX_SECONDS) * C.SR))


# ------------------------------------------------------------------ features
def logmel(y):
    m = librosa.feature.melspectrogram(y=y, sr=C.SR, n_fft=C.N_FFT, win_length=C.WIN_LENGTH,
                                       hop_length=C.HOP_LENGTH, n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX)
    return librosa.power_to_db(m, ref=np.max).astype(np.float32)


def mfcc_deltas(y, n_mfcc=C.N_MFCC):
    m = librosa.feature.mfcc(y=y, sr=C.SR, n_mfcc=n_mfcc, n_fft=C.N_FFT, win_length=C.WIN_LENGTH,
                             hop_length=C.HOP_LENGTH, n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX)
    width = min(9, m.shape[1] if m.shape[1] % 2 else m.shape[1] - 1)
    if width < 3:
        d1 = d2 = np.zeros_like(m)
    else:
        d1 = librosa.feature.delta(m, width=width)
        d2 = librosa.feature.delta(m, order=2, width=width)
    return np.vstack([m, d1, d2]).astype(np.float32)


def summary_stats(y):
    """Fixed-length, ORDER-FREE summary of a clip (baseline representation)."""
    m = librosa.feature.mfcc(y=y, sr=C.SR, n_mfcc=20, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH)
    d = librosa.feature.delta(m, width=min(9, m.shape[1] if m.shape[1] % 2 else m.shape[1] - 1)) \
        if m.shape[1] >= 3 else np.zeros_like(m)
    spec = [librosa.feature.spectral_centroid(y=y, sr=C.SR, hop_length=C.HOP_LENGTH),
            librosa.feature.spectral_bandwidth(y=y, sr=C.SR, hop_length=C.HOP_LENGTH),
            librosa.feature.spectral_rolloff(y=y, sr=C.SR, hop_length=C.HOP_LENGTH),
            librosa.feature.zero_crossing_rate(y, hop_length=C.HOP_LENGTH),
            librosa.feature.rms(y=y, hop_length=C.HOP_LENGTH)]
    spec = np.vstack(spec)
    feats = [m.mean(1), m.std(1), m.min(1), m.max(1), d.mean(1), d.std(1),
             spec.mean(1), spec.std(1), [len(y) / C.SR]]
    return np.concatenate([np.ravel(f) for f in feats]).astype(np.float32)


def _frames(seconds=None):
    return 1 + n_samples(seconds) // C.HOP_LENGTH


def extract(path, kind, trim=True, seconds=None):
    """One clip -> one representation. `kind` in {wave, logmel, mfcc, mfcc_var, stats}."""
    y = load_audio(path, trim=trim)
    if kind == "stats":
        return summary_stats(y)
    if kind == "mfcc_var":                      # variable length, for DTW
        return mfcc_deltas(y)
    y = fix_length(y, n_samples(seconds))
    if kind == "wave":
        return y
    if kind == "logmel":
        return logmel(y)[:, :_frames(seconds)]
    if kind == "mfcc":
        return mfcc_deltas(y)[:, :_frames(seconds)]
    raise ValueError(kind)


def compute_features(paths, kind, trim=True, seconds=None, n_jobs=-1, cache=True):
    """Extract features for many files in parallel, with an on-disk cache."""
    paths = list(map(str, paths))
    key = json.dumps({"kind": kind, "trim": trim, "sec": seconds or C.MAX_SECONDS, "sr": C.SR,
                      "mels": C.N_MELS, "mfcc": C.N_MFCC, "hop": C.HOP_LENGTH, "top_db": C.TOP_DB,
                      "files": [Path(p).name for p in paths]}, sort_keys=True)
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_file = Path(C.CACHE_DIR) / f"{kind}_{h}.joblib"
    if cache and cache_file.exists():
        return joblib.load(cache_file)
    out = Parallel(n_jobs=n_jobs)(delayed(extract)(p, kind, trim, seconds)
                                  for p in tqdm(paths, desc=f"{kind} features"))
    out = out if kind == "mfcc_var" else np.stack(out)
    if cache:
        joblib.dump(out, cache_file)
    return out


class Standardizer:
    """Per-feature-row mean/std fitted on TRAIN only (avoids leakage), for (N, F, T) arrays."""

    def fit(self, X):
        self.mean = X.mean(axis=(0, 2), keepdims=True)
        self.std = X.std(axis=(0, 2), keepdims=True) + 1e-6
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)


# ------------------------------------------------------------------ EDA statistics
def audio_stats(path):
    """Per-file descriptive statistics used in EDA and later in error analysis."""
    info = sf.info(path)
    raw, sr = librosa.load(path, sr=None, mono=True)
    y = librosa.resample(raw, orig_sr=sr, target_sr=C.SR) if sr != C.SR else raw
    dur = len(y) / C.SR
    _, (s, e) = librosa.effects.trim(y, top_db=C.TOP_DB) if y.size else (None, (0, 0))
    rms = librosa.feature.rms(y=y, hop_length=C.HOP_LENGTH)[0] if y.size else np.array([0.0])
    rms_db = 20 * np.log10(rms + 1e-10)
    lo, hi = np.percentile(rms_db, [10, 90])
    with open(path, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()
    return {
        "path": str(path), "orig_sr": info.samplerate, "channels": info.channels, "format": info.format,
        "duration_s": dur, "trimmed_duration_s": (e - s) / C.SR,
        "lead_silence_s": s / C.SR, "trail_silence_s": (len(y) - e) / C.SR,
        "silence_ratio": 1 - (e - s) / max(len(y), 1),
        "rms_db": float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-10)) if y.size else -200.0,
        "peak": float(np.max(np.abs(raw))) if raw.size else 0.0,
        "clip_frac": float(np.mean(np.abs(raw) > 0.99)) if raw.size else 0.0,
        "snr_est_db": float(hi - lo),           # loud-frame vs quiet-frame energy gap (rough SNR proxy)
        "md5": md5,
    }


def compute_audio_stats(paths, n_jobs=-1):
    import pandas as pd
    rows = Parallel(n_jobs=n_jobs)(delayed(audio_stats)(p) for p in tqdm(list(paths), desc="audio stats"))
    return pd.DataFrame(rows)
