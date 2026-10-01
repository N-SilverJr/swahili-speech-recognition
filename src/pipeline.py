"""Experiment runner shared by the three neural-model notebooks.

Each experiment is a dict:
    {"id": "CNN-02", "description": "...", "hypothesis": "...", "motivated_by": "CNN-01: ...",
     "params": {...model/data options...}, "train": {...fit() options...}}
Every run is seeded, logged to experiments/log_<model>.csv, and the best configuration by
VALIDATION log loss is kept. The test split is only used afterwards, once, in finalize().
"""
import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as C
from .evaluate import compute_metrics, log_experiment, make_submission, save_final_results
from .models import count_parameters
from .train import balanced_class_weights, fit, predict_proba
from .utils import Timer, set_seed


def run_experiments(experiments, build_model, make_datasets, *, member, model_name, n_classes,
                    y_train=None, default_train=None):
    """build_model(params) -> nn.Module ; make_datasets(params) -> (train_ds, val_ds)."""
    best, rows = None, []
    for e in experiments:
        print(f"\n===== {e['id']}: {e['description']} =====")
        set_seed(C.SEED)
        train_ds, val_ds = make_datasets(e["params"])
        model = build_model(e["params"])
        tkw = {**(default_train or {}), **e.get("train", {})}
        if tkw.pop("class_weights", False) and y_train is not None:
            tkw["class_weights"] = balanced_class_weights(y_train, n_classes)
        if C.QUICK:
            tkw["epochs"], tkw["patience"] = min(tkw.get("epochs", 2), 2), 2
        n_params = count_parameters(model)
        with Timer() as t:
            model, hist = fit(model, train_ds, val_ds, **tkw)
        pv, yv = predict_proba(model, DataLoader(val_ds, batch_size=128))
        m = compute_metrics(yv, pv, n_classes)
        log_experiment(e["id"], member, model_name, e["description"], e["hypothesis"],
                       {"params": e["params"], "train": {k: v for k, v in tkw.items() if k != "class_weights"}},
                       m, e.get("motivated_by", ""), t.seconds, n_params)
        rows.append({"exp_id": e["id"], "description": e["description"], **{f"val_{k}": v for k, v in m.items()},
                     "best_epoch": int(hist.loc[hist.val_loss.idxmin(), "epoch"]), "epochs_run": len(hist),
                     "train_gap_acc": float(hist.train_acc.iloc[-1] - hist.val_acc.iloc[-1]),
                     "params": n_params, "minutes": t.seconds / 60})
        print(f"--> {e['id']} val log loss {m['log_loss']:.4f} | acc {m['accuracy']:.3f} | "
              f"macro-F1 {m['macro_f1']:.3f} | {t.seconds/60:.1f} min")
        if best is None or m["log_loss"] < best["metrics"]["log_loss"]:
            best = {"exp": e, "state": {k: v.cpu() for k, v in model.state_dict().items()},
                    "history": hist, "metrics": m, "n_params": n_params}
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = pd.DataFrame(rows)
    print(f"\nBest configuration on validation: {best['exp']['id']} ({best['exp']['description']})")
    return best, summary


def finalize(best, build_model, test_ds, y_test, test_files, classes, *, key, display_name,
             zindi_ds=None, zindi_df=None):
    """Evaluate the selected model ONCE on the held-out test split and write a Zindi submission."""
    model = build_model(best["exp"]["params"])
    model.load_state_dict(best["state"])
    pt, yt = predict_proba(model, DataLoader(test_ds, batch_size=128))
    assert np.array_equal(yt, y_test)
    metrics = save_final_results(key, display_name, y_test, pt, classes, test_files, best["history"],
                                 extra={"selected_experiment": best["exp"]["id"],
                                        "n_trainable_params": best["n_params"],
                                        "val_log_loss": best["metrics"]["log_loss"]})
    torch.save(best["state"], C.RESULTS_DIR / key / "model_state.pt")
    if zindi_ds is not None:
        pz, _ = predict_proba(model, DataLoader(zindi_ds, batch_size=128))
        make_submission(zindi_df, pz, classes, key)
    return model, pt, metrics
