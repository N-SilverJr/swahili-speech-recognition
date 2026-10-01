"""Fill the {{placeholders}} in report/report_draft.md with numbers from the results folders.

    python report/fill_report.py          ->  report/report_filled.md

Run it after notebooks 01-06. Any placeholder whose source file does not exist yet is left
as-is and listed at the end, so you can see what is still missing.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and k not in ("class_counts", "orig_sample_rates", "channels", "most_similar_pairs"):
            out.update(flatten(v, key + "."))
        out[key] = v
    return out


def fmt(key, v):
    if key in ("frac_flagged", "silence_ratio_median"):
        return f"{100 * float(v):.1f}%"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) >= 1 else f"{v:.3f}"
    if isinstance(v, list):
        return ", ".join(f"*{x}*" for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k} ({val})" for k, val in v.items())
    return str(v)


values = {"max_seconds": C.MAX_SECONDS, "n_frames": int(C.MAX_SECONDS * C.SR) // C.HOP_LENGTH + 1,
          "n_samples": f"{int(C.MAX_SECONDS * C.SR):,}"}

eda = C.RESULTS_DIR / "eda" / "eda_summary.json"
if eda.exists():
    s = json.loads(eda.read_text())
    values.update(flatten(s))
    if "class_counts" in s:
        values["class_count_min"] = min(s["class_counts"].values())
        values["class_count_max"] = max(s["class_counts"].values())

logs = list((ROOT / "experiments").glob("log_*.csv"))
if logs:
    import pandas as pd
    values["n_experiments"] = sum(len(pd.read_csv(f)) for f in logs)

table = C.RESULTS_DIR / "comparison" / "comparison_table.md"
if table.exists():
    values["comparison_table"] = table.read_text()

ens = C.RESULTS_DIR / "comparison" / "ensemble_exploratory.json"
if ens.exists():
    values["ensemble_log_loss"] = json.loads(ens.read_text())["log_loss"]

text = (ROOT / "report" / "report_draft.md").read_text()
missing = set()


def repl(m):
    k = m.group(1)
    if k in values:
        return fmt(k, values[k])
    missing.add(k)
    return m.group(0)


out = re.sub(r"\{\{([\w.%]+)\}\}", repl, text)
(ROOT / "report" / "report_filled.md").write_text(out)
print("Wrote report/report_filled.md")
if missing:
    print("Still missing (run the notebooks that produce them):", sorted(missing))
print(f"{out.count('[[WRITE')} [[WRITE ...]] paragraphs still need to be written by the group.")
