"""Shared evaluation: every model is scored and plotted with exactly the same code."""
import json
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             log_loss, roc_auc_score, roc_curve, top_k_accuracy_score)
from sklearn.preprocessing import label_binarize

from . import config as C

EPS = 1e-7


def _safe(proba):
    p = np.clip(proba, EPS, 1)
    return p / p.sum(1, keepdims=True)


# ------------------------------------------------------------------ metrics
def expected_calibration_error(y, proba, n_bins=15):
    conf, pred = proba.max(1), proba.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y, proba, n_classes=None):
    n_classes = n_classes or proba.shape[1]
    labels = list(range(n_classes))
    pred = proba.argmax(1)
    out = {
        "log_loss": log_loss(y, _safe(proba), labels=labels),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", labels=labels, zero_division=0),
        "top2_accuracy": top_k_accuracy_score(y, proba, k=2, labels=labels) if n_classes > 2 else np.nan,
        "ece": expected_calibration_error(y, proba),
    }
    try:
        out["roc_auc_ovr_macro"] = roc_auc_score(label_binarize(y, classes=labels), proba, average="macro")
    except ValueError:
        out["roc_auc_ovr_macro"] = np.nan
    return {k: float(v) for k, v in out.items()}


def per_class_table(y, proba, classes):
    rep = classification_report(y, proba.argmax(1), labels=list(range(len(classes))),
                                target_names=classes, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.loc[classes, ["precision", "recall", "f1-score", "support"]]


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test: are two models' error patterns significantly different?"""
    a_right, b_right = pred_a == y, pred_b == y
    b01, b10 = int((a_right & ~b_right).sum()), int((~a_right & b_right).sum())
    p = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return {"a_right_b_wrong": b01, "a_wrong_b_right": b10, "p_value": float(p)}


# ------------------------------------------------------------------ plots
def plot_learning_curves(hist, title, path=None):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].plot(hist.epoch, hist.train_loss, label="train"); ax[0].plot(hist.epoch, hist.val_loss, label="val")
    ax[0].set_title("Cross-entropy / log loss"); ax[0].legend()
    ax[1].plot(hist.epoch, hist.train_acc, label="train"); ax[1].plot(hist.epoch, hist.val_acc, label="val")
    ax[1].set_title("Accuracy"); ax[1].legend()
    ax[2].plot(hist.epoch, hist.val_macro_f1, color="C2"); ax[2].set_title("Validation macro-F1")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    fig.suptitle(title); fig.tight_layout()
    _save(fig, path)


def plot_confusion(y, pred, classes, title, path=None):
    cm = confusion_matrix(y, pred, labels=list(range(len(classes))), normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes,
                vmin=0, vmax=1, ax=ax, annot_kws={"size": 7})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.xticks(rotation=45, ha="right"); fig.tight_layout()
    _save(fig, path)


def plot_roc(y, proba, classes, title, path=None):
    Y = label_binarize(y, classes=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, c in enumerate(classes):
        if Y[:, i].any():
            fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
            ax.plot(fpr, tpr, lw=1, label=f"{c} (AUC {roc_auc_score(Y[:, i], proba[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout()
    _save(fig, path)


def plot_reliability(y, proba, title, path=None, n_bins=10):
    conf, correct = proba.max(1), proba.argmax(1) == y
    bins = np.linspace(0, 1, n_bins + 1)
    mids, accs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            mids.append(conf[m].mean()); accs.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=.8, label="perfect calibration")
    ax.plot(mids, accs, "o-", label=f"model (ECE {expected_calibration_error(y, proba):.3f})")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------ experiment log
LOG_COLS = ["exp_id", "timestamp", "member", "model", "description", "hypothesis", "motivated_by",
            "config", "val_log_loss", "val_accuracy", "val_macro_f1", "train_seconds", "n_trainable_params",
            "observation"]


def log_experiment(exp_id, member, model, description, hypothesis, config, val_metrics,
                   motivated_by="", train_seconds=None, n_params=None, observation=""):
    """Append (or overwrite, if re-run) one row of experiments/experiment_log.csv.

    `observation` is for the member to fill in by hand after looking at the result:
    what happened, why you think it happened, and what you tried next because of it.
    """
    row = {"exp_id": exp_id, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "member": member,
           "model": model, "description": description, "hypothesis": hypothesis,
           "motivated_by": motivated_by, "config": json.dumps(config, default=str),
           "val_log_loss": round(val_metrics["log_loss"], 4), "val_accuracy": round(val_metrics["accuracy"], 4),
           "val_macro_f1": round(val_metrics["macro_f1"], 4),
           "train_seconds": None if train_seconds is None else round(train_seconds, 1),
           "n_trainable_params": n_params, "observation": observation}
    path = _log_path(model)
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LOG_COLS)
    if exp_id in set(log.exp_id):
        keep_obs = log.loc[log.exp_id == exp_id, "observation"].iloc[0]
        if not observation and isinstance(keep_obs, str):
            row["observation"] = keep_obs          # don't wipe a hand-written observation on re-run
        log = log[log.exp_id != exp_id]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True).sort_values("exp_id")
    log.to_csv(path, index=False)
    return row


def _log_path(model):
    # one log file per model so members working in parallel never overwrite each other
    slug = "".join(ch if ch.isalnum() else "_" for ch in model.lower()).strip("_")
    return C.EXP_LOG.parent / f"log_{slug}.csv"


def show_log(model=None):
    """All experiments (optionally one model). Also rebuilds experiments/experiment_log.csv."""
    files = sorted(C.EXP_LOG.parent.glob("log_*.csv"))
    if not files:
        return pd.DataFrame(columns=LOG_COLS)
    log = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).sort_values("exp_id")
    log.to_csv(C.EXP_LOG, index=False)
    return log[log.model == model].reset_index(drop=True) if model else log.reset_index(drop=True)


# ------------------------------------------------------------------ final test evaluation
def save_final_results(key, display_name, y, proba, classes, files, history=None, extra=None):
    """Evaluate the SELECTED configuration once on the held-out test split and save
    everything the report and the comparison notebook need under results/<key>/."""
    out = C.RESULTS_DIR / key
    out.mkdir(parents=True, exist_ok=True)
    m = compute_metrics(y, proba, len(classes))
    m.update({"model": display_name, "key": key, **(extra or {})})
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    pred = proba.argmax(1)
    df = pd.DataFrame({"file": files, "true": [classes[i] for i in y], "pred": [classes[i] for i in pred],
                       "confidence": proba.max(1), "p_true": proba[np.arange(len(y)), y]})
    for i, c in enumerate(classes):
        df[f"p_{c}"] = proba[:, i]
    df.to_csv(out / "predictions.csv", index=False)
    per_class_table(y, proba, classes).to_csv(out / "per_class.csv")
    plot_confusion(y, pred, classes, f"{display_name} - normalised confusion matrix (test)", out / "confusion.png")
    plot_roc(y, proba, classes, f"{display_name} - one-vs-rest ROC (test)", out / "roc.png")
    plot_reliability(y, proba, f"{display_name} - reliability", out / "reliability.png")
    if history is not None:
        history.to_csv(out / "history.csv", index=False)
        plot_learning_curves(history, display_name, out / "learning_curves.png")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=2))
    return m


def make_submission(test_df, proba, classes, name):
    """Write a Zindi submission (one probability column per word)."""
    sample = C.DATA_DIR / "SampleSubmission.csv"
    id_col = test_df.attrs.get("id_col", "AUDIO")
    sub = pd.DataFrame(proba, columns=classes)
    sub.insert(0, id_col, test_df["file"].values)
    if sample.exists():
        s = pd.read_csv(sample)
        id_col = s.columns[0]
        sub = sub.rename(columns={sub.columns[0]: id_col})
        cols = [c for c in s.columns[1:] if c in sub.columns]
        if len(cols) == len(classes):
            sub = sub[[id_col] + cols]
    path = C.SUBMISSION_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"Submission written to {path}")
    return path
