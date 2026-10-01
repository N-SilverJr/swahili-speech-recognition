"""Central configuration shared by every notebook.

Every member imports from here so that all five approaches use the SAME audio
settings, the SAME data split and the SAME evaluation code. Change a value here
(not inside a notebook) and record the change in the experiment log.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
# Folder holding Train.csv, Test.csv, SampleSubmission.csv and the unzipped Audio.zip.
# On Colab this is set to a Google Drive folder in the setup cell of each notebook.
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", DATA_DIR))       # searched recursively for audio files
CACHE_DIR = Path(os.environ.get("CACHE_DIR", DATA_DIR / "cache"))
SPLITS_PATH = ROOT / "data" / "splits.csv"                    # committed to git so all members share it
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
EXP_LOG = ROOT / "experiments" / "experiment_log.csv"
SUBMISSION_DIR = ROOT / "submissions"

# Optional overrides if automatic column detection picks the wrong column
AUDIO_COL = os.environ.get("AUDIO_COL")
LABEL_COL = os.environ.get("LABEL_COL")

# ---------------------------------------------------------------- audio
SR = 16_000            # all audio resampled to 16 kHz (required by wav2vec 2.0 / XLS-R)
TOP_DB = 30            # silence-trimming threshold (dB below peak)
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", 1.5))  # fixed input length; set from EDA (~99th pct of trimmed duration)
N_FFT = 512
WIN_LENGTH = 400       # 25 ms analysis window
HOP_LENGTH = 160       # 10 ms hop -> 100 frames per second
N_MELS = 64
N_MFCC = 13            # + delta + delta-delta = 39 features per frame
FMIN, FMAX = 20, 8000

# ---------------------------------------------------------------- experiment
SEED = 42
VAL_SIZE = 0.10        # fraction of labelled data used for validation / model selection
TEST_SIZE = 0.10       # held-out test split, evaluated ONCE per model at the very end
QUICK = os.environ.get("QUICK") == "1"   # smoke-test mode: tiny epochs, only to check the code runs

for _d in (CACHE_DIR, RESULTS_DIR, FIG_DIR, EXP_LOG.parent, SUBMISSION_DIR, SPLITS_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)
