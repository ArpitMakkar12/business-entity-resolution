"""Central configuration: paths and global constants.

Every path can be overridden with an environment variable, so the same code
runs unchanged on a laptop, on Colab and in the reviewers' environment:

    BER_DATA_DIR    folder containing train/ and test/   (default: ./dataset)
    BER_WORK_DIR    intermediate artefacts (parquet, models, reports)
    BER_OUTPUT_DIR  final matching_results.tsv / candidate_pairs.tsv
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", PROJECT_ROOT / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", PROJECT_ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", PROJECT_ROOT / "output"))

SEED = 42
N_JOBS = int(os.environ.get("BER_N_JOBS", os.cpu_count() or 4))


def train_paths(data_dir: Path = None) -> dict:
    """Return the four training file paths (source1/2/3 + ground truth)."""
    d = Path(data_dir or DATA_DIR) / "train"
    return {
        "s1": d / "train_source1.tsv",
        "s2": d / "train_source2.tsv",
        "s3": d / "train_source3.tsv",
        "gt": d / "train_ground_truth.tsv",
    }


def test_paths(data_dir: Path = None) -> dict:
    """Return the three test file paths (source1/2/3)."""
    d = Path(data_dir or DATA_DIR) / "test"
    return {
        "s1": d / "test_source1.tsv",
        "s2": d / "test_source2.tsv",
        "s3": d / "test_source3.tsv",
    }
