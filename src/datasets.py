"""PyTorch datasets and data augmentation."""
import numpy as np
import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    """Wraps a pre-computed feature array (N, F, T) or waveform array (N, samples)."""

    def __init__(self, X, y=None, transform=None):
        self.X, self.y, self.transform = X, y, transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.X[i], dtype=np.float32))
        if self.transform is not None:
            x = self.transform(x)
        y = -1 if self.y is None else int(self.y[i])
        return x, y


class SpecAugment:
    """Time shift + frequency/time masking on a (F, T) feature map (Park et al., 2019).

    Speakers start the word at different moments and recordings differ in quality, so
    shifting and masking teach the model to rely on the pattern, not its exact position.
    """

    def __init__(self, max_shift=0.1, freq_masks=2, freq_width=8, time_masks=2, time_width=0.1, p=0.8):
        self.max_shift, self.fm, self.fw, self.tm, self.tw, self.p = \
            max_shift, freq_masks, freq_width, time_masks, time_width, p

    def __call__(self, x):
        if np.random.rand() > self.p:
            return x
        x = x.clone()
        F_, T = x.shape
        s = int(np.random.uniform(-self.max_shift, self.max_shift) * T)
        if s:
            x = torch.roll(x, s, dims=1)
            if s > 0:
                x[:, :s] = 0
            else:
                x[:, s:] = 0
        for _ in range(self.fm):
            w = np.random.randint(0, self.fw + 1)
            f0 = np.random.randint(0, max(1, F_ - w))
            x[f0:f0 + w, :] = 0
        tw = max(1, int(self.tw * T))
        for _ in range(self.tm):
            w = np.random.randint(0, tw + 1)
            t0 = np.random.randint(0, max(1, T - w))
            x[:, t0:t0 + w] = 0
        return x


class WaveAugment:
    """Waveform augmentation for wav2vec: random gain, time shift and additive noise."""

    def __init__(self, max_shift=0.1, snr_db=(10, 30), gain_db=(-6, 6), p=0.8):
        self.max_shift, self.snr_db, self.gain_db, self.p = max_shift, snr_db, gain_db, p

    def __call__(self, x):
        if np.random.rand() > self.p:
            return x
        n = x.shape[0]
        x = torch.roll(x, int(np.random.uniform(-self.max_shift, self.max_shift) * n))
        x = x * 10 ** (np.random.uniform(*self.gain_db) / 20)
        power = x.pow(2).mean().clamp_min(1e-10)
        snr = 10 ** (np.random.uniform(*self.snr_db) / 10)
        return x + torch.randn_like(x) * torch.sqrt(power / snr)


class Normalize:
    """Per-clip zero-mean / unit-variance (what wav2vec 2.0 feature extractors do)."""

    def __init__(self, inner=None):
        self.inner = inner

    def __call__(self, x):
        if self.inner is not None:
            x = self.inner(x)
        return (x - x.mean()) / (x.std() + 1e-7)
