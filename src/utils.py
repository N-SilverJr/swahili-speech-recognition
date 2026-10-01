import os
import random
import time

import numpy as np


def set_seed(seed: int = 42):
    """Seed python, numpy and torch so every run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0


def persist_outputs():
    """Colab sessions are wiped when they end. Copy results/, experiments/ and submissions/
    to Google Drive (OUTPUT_BACKUP_DIR) so nothing is lost; then upload them to GitHub."""
    import shutil
    from pathlib import Path
    from . import config as C
    dest = os.environ.get("OUTPUT_BACKUP_DIR")
    if not dest:
        print("OUTPUT_BACKUP_DIR not set (running locally?) - outputs are already in the repo folder.")
        return
    for sub in ("results", "experiments", "submissions"):
        src = C.ROOT / sub
        if src.exists():
            shutil.copytree(src, Path(dest) / sub, dirs_exist_ok=True)
    print(f"Outputs copied to {dest}")
