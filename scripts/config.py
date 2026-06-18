import os
from pathlib import Path


def get_dataset_dir():
    dataset_dir = os.environ.get("SPINAL_AI_DATASET_DIR")

    if dataset_dir:
        return Path(dataset_dir).expanduser()

    return Path(__file__).resolve().parents[1]
