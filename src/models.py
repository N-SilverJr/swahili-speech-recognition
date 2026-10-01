"""The three neural sequential architectures compared in the study.

1. LogMelCNN        - 2-D CNN over the log-mel spectrogram: learns LOCAL time-frequency
                      patterns (formant transitions, onsets) with translation invariance.
2. BiRNNAttention   - bidirectional GRU/LSTM over MFCC frames: models the ORDER of
                      acoustic events across the whole utterance; attention pooling learns
                      which frames matter (and ignores silence/padding).
3. HFAudioClassifier- pretrained self-supervised transformer (wav2vec 2.0 / XLS-R / MMS)
                      on the raw waveform: global self-attention + representations
                      learned from large unlabelled (multilingual) speech corpora.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ 1. CNN
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class LogMelCNN(nn.Module):
    def __init__(self, n_classes, channels=(32, 64, 128, 256), dropout=0.2, head_dropout=0.3):
        super().__init__()
        layers, c = [], 1
        for ch in channels:
            layers.append(ConvBlock(c, ch, dropout))
            c = ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(c, n_classes))

    def forward(self, x):                      # x: (B, n_mels, T)
        x = self.features(x.unsqueeze(1))      # (B, C, F', T')
        x = x.mean(dim=2)                      # average over frequency
        x = x.amax(dim=-1) + x.mean(dim=-1)    # max + mean over time
        return self.head(x)


# ------------------------------------------------------------------ 2. BiRNN + attention
class AttentionPool(nn.Module):
    """Additive (Bahdanau-style) attention pooling over time steps."""

    def __init__(self, d, attn_dim=128):
        super().__init__()
        self.proj = nn.Linear(d, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h):                      # h: (B, T, d)
        a = torch.softmax(self.v(torch.tanh(self.proj(h))).squeeze(-1), dim=1)
        return (a.unsqueeze(-1) * h).sum(1), a


class BiRNNAttention(nn.Module):
    def __init__(self, n_features, n_classes, hidden=128, layers=2, rnn="gru",
                 pooling="attention", bidirectional=True, dropout=0.3):
        super().__init__()
        RNN = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn]
        self.rnn = RNN(n_features, hidden, num_layers=layers, batch_first=True,
                       bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        d = hidden * (2 if bidirectional else 1)
        self.pooling = pooling
        self.attn = AttentionPool(d) if pooling == "attention" else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, d // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d // 2, n_classes))
        self.last_attention = None

    def forward(self, x):                      # x: (B, F, T)
        h, _ = self.rnn(x.transpose(1, 2))     # (B, T, d)
        if self.pooling == "attention":
            z, a = self.attn(h)
            self.last_attention = a.detach()
        elif self.pooling == "mean":
            z = h.mean(1)
        else:                                  # "last": final time step (forward dir) + first (backward dir)
            half = h.shape[-1] // 2
            z = torch.cat([h[:, -1, :half], h[:, 0, half:]], dim=-1) if self.rnn.bidirectional else h[:, -1]
        return self.head(z)


# ------------------------------------------------------------------ 3. pretrained transformer
class HFAudioClassifier(nn.Module):
    """Wrapper around a Hugging Face speech encoder with a classification head.

    freeze="feature_encoder" : freeze the CNN waveform encoder, fine-tune the transformer (standard)
    freeze="base"            : freeze the whole pretrained model, train only the head (linear probe)
    freeze="none"            : fine-tune everything
    """

    def __init__(self, model_name, n_classes, freeze="feature_encoder", tiny_random=False):
        super().__init__()
        if tiny_random:                        # only for offline smoke tests - never for real experiments
            from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification
            cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                 intermediate_size=64, conv_dim=(16, 16), conv_stride=(5, 4),
                                 conv_kernel=(10, 4), num_conv_pos_embeddings=16,
                                 num_conv_pos_embedding_groups=2, num_labels=n_classes,
                                 classifier_proj_size=16)
            self.model = Wav2Vec2ForSequenceClassification(cfg)
        else:
            from transformers import AutoModelForAudioClassification
            self.model = AutoModelForAudioClassification.from_pretrained(
                model_name, num_labels=n_classes, ignore_mismatched_sizes=True)
        if freeze in ("feature_encoder", "base") and hasattr(self.model, "freeze_feature_encoder"):
            self.model.freeze_feature_encoder()
        if freeze == "base":
            for p in self.model.base_model.parameters():
                p.requires_grad = False

    def forward(self, x):                      # x: (B, samples), zero-mean / unit-variance
        return self.model(input_values=x).logits


def count_parameters(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
