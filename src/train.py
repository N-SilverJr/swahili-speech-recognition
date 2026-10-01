"""One training loop shared by the CNN, BiRNN and transformer, so differences in results
come from the architectures and representations, not from different training code."""
import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from .evaluate import _safe
from torch.utils.data import DataLoader

from .utils import get_device


@torch.no_grad()
def predict_proba(model, loader, device=None, amp=True):
    device = device or get_device()
    model.eval().to(device)
    probs, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(ys)


def fit(model, train_ds, val_ds, *, epochs=40, lr=1e-3, batch_size=64, weight_decay=1e-4,
        patience=8, class_weights=None, label_smoothing=0.0, grad_accum=1, amp=True,
        num_workers=2, device=None, verbose=True):
    """Train with AdamW + one-cycle LR, early stopping on validation LOG LOSS (the official
    Zindi metric). Returns (model restored to its best epoch, history DataFrame)."""
    device = device or get_device()
    model.to(device)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    drop_last=len(train_ds) > batch_size)
    vl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(tl) / grad_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch,
                                                pct_start=0.1)
    w = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_classes = None

    hist, best, best_state, bad = [], np.inf, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, correct, n = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tl):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = crit(logits.float(), y) / grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(tl):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch + 1 < sched.total_steps:
                    sched.step()
            tot += loss.item() * grad_accum * len(y)
            correct += (logits.argmax(-1) == y).sum().item()
            n += len(y)
            n_classes = logits.shape[-1]
        pv, yv = predict_proba(model, vl, device, amp)
        row = {"epoch": ep, "train_loss": tot / n, "train_acc": correct / n,
               "val_loss": log_loss(yv, _safe(pv), labels=list(range(n_classes))),
               "val_acc": accuracy_score(yv, pv.argmax(1)),
               "val_macro_f1": f1_score(yv, pv.argmax(1), average="macro"),
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        hist.append(row)
        if verbose:
            print(f"ep {ep:3d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} | "
                  f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.3f} F1 {row['val_macro_f1']:.3f} | "
                  f"{row['seconds']:.0f}s")
        if row["val_loss"] < best - 1e-4:
            best, bad = row["val_loss"], 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"Early stopping at epoch {ep} (best val log loss {best:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)


def balanced_class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return len(y) / (n_classes * np.maximum(counts, 1))
