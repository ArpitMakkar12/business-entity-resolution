"""Score a matching_results.tsv against the hidden synthetic test truth (smoke tests only)."""
import sys

import polars as pl

from src.decision import f05_per_entity
from src.io_utils import read_ground_truth

truth = read_ground_truth(sys.argv[1])
pred = read_ground_truth(sys.argv[2]).rename({"matched": "ids"}).drop("n_matches")
per = f05_per_entity(pred, truth)
print(f"synthetic TEST macro F0.5 = {per['f05'].mean():.5f}  "
      f"(precision {per['precision'].mean():.4f}, recall {per['recall'].mean():.4f})")
