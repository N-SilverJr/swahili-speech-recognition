"""Loading the Zindi CSVs, locating audio files and building the shared split."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C

AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm")


# ------------------------------------------------------------------ CSV parsing
def _audio_column(df):
    if C.AUDIO_COL:
        return C.AUDIO_COL
    for c in df.columns:
        s = df[c].astype(str).str.lower()
        if s.str.endswith(AUDIO_EXTS).mean() > 0.5:
            return c
    for c in df.columns:
        if any(k in c.lower() for k in ("audio", "file", "path", "id")):
            return c
    raise ValueError(f"Could not find the audio column in {list(df.columns)}; set AUDIO_COL.")


def _label_info(df, audio_col):
    """Return (label_col, None) for long format, or (None, one_hot_cols) for wide format."""
    if C.LABEL_COL:
        return C.LABEL_COL, None
    others = [c for c in df.columns if c != audio_col]
    text = [c for c in others if not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique() <= 200]
    if text:
        return text[0], None
    onehot = [c for c in others if pd.api.types.is_numeric_dtype(df[c])
              and set(pd.unique(df[c].dropna())) <= {0, 1}]
    if len(onehot) >= 2:
        return None, onehot
    raise ValueError(f"Could not find the label column(s) in {list(df.columns)}; set LABEL_COL.")


def load_labelled(csv_path=None) -> pd.DataFrame:
    """Train.csv -> DataFrame with columns [file, label]."""
    csv_path = Path(csv_path or C.DATA_DIR / "Train.csv")
    raw = pd.read_csv(csv_path)
    a = _audio_column(raw)
    label_col, onehot = _label_info(raw, a)
    labels = raw[label_col].astype(str).str.strip() if label_col else raw[onehot].idxmax(axis=1)
    df = pd.DataFrame({"file": raw[a].astype(str).str.strip(), "label": labels})
    print(f"Loaded {len(df)} labelled clips | audio column = '{a}' | "
          f"labels from {'column ' + repr(label_col) if label_col else 'one-hot columns'} | "
          f"{df.label.nunique()} classes")
    return df


def load_unlabelled(csv_path=None) -> pd.DataFrame:
    """Zindi Test.csv (no labels) -> DataFrame [file]; original id column kept in .attrs."""
    csv_path = Path(csv_path or C.DATA_DIR / "Test.csv")
    raw = pd.read_csv(csv_path)
    a = _audio_column(raw)
    df = pd.DataFrame({"file": raw[a].astype(str).str.strip()})
    df.attrs["id_col"] = a
    return df


# ------------------------------------------------------------------ audio paths
_INDEX = None


def audio_index(audio_dir=None):
    """Map file name AND stem -> full path (Audio.zip may contain sub-folders)."""
    global _INDEX
    if _INDEX is None:
        root = Path(audio_dir or C.AUDIO_DIR)
        idx = {}
        for p in root.rglob("*"):
            if p.suffix.lower() in AUDIO_EXTS and "cache" not in p.parts:
                idx.setdefault(p.name, p)
                idx.setdefault(p.stem, p)
        _INDEX = idx
        print(f"Indexed {len(set(idx.values()))} audio files under {root}")
    return _INDEX


def add_paths(df: pd.DataFrame) -> pd.DataFrame:
    idx = audio_index()
    df = df.copy()
    df["path"] = [idx.get(Path(f).name) or idx.get(Path(f).stem) for f in df["file"]]
    missing = int(df["path"].isna().sum())
    if missing:
        print(f"WARNING: {missing} files listed in the CSV were not found on disk and are dropped.")
        df = df.dropna(subset=["path"])
    df["path"] = df["path"].astype(str)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ shared split
def get_splits(rebuild: bool = False) -> pd.DataFrame:
    """Stratified train / val / test split saved to data/splits.csv.

    Member 1 creates it once and commits it; everyone else loads the same file, so all
    five approaches are compared on identical data. Zindi's Test.csv has no labels, so
    our labelled 'test' split is carved out of Train.csv.
    """
    if C.SPLITS_PATH.exists() and not rebuild:
        df = pd.read_csv(C.SPLITS_PATH)
    else:
        df = load_labelled()
        trval, test = train_test_split(df, test_size=C.TEST_SIZE, stratify=df.label, random_state=C.SEED)
        val_frac = C.VAL_SIZE / (1 - C.TEST_SIZE)
        train, val = train_test_split(trval, test_size=val_frac, stratify=trval.label, random_state=C.SEED)
        df = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
        df[["file", "label", "split"]].to_csv(C.SPLITS_PATH, index=False)
        print(f"Saved new split to {C.SPLITS_PATH} (commit this file!)")
    df = add_paths(df)
    print("Split sizes:", df.split.value_counts().to_dict())
    return df


def label_encoder(labels):
    classes = sorted(pd.unique(pd.Series(labels)))
    return classes, {c: i for i, c in enumerate(classes)}


def encode(labels, to_id):
    return np.array([to_id[l] for l in labels], dtype=np.int64)
