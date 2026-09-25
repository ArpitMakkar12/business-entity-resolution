"""Compare house-number conflict rules on validation and test.

For every variant in src.postprocess.VARIANTS (and a few stricter thresholds
for reference) it reports:
  validation  macro F0.5 and how many true / false predicted pairs the rule removes
  test        how many predicted pairs the rule removes, and predicted matches
              per S1 per country after the rule
The validation cost of a rule is exact; its benefit on test cannot be measured
without labels, but the audit showed the removed pattern is mostly false merges.

Usage:  python -m src.tune_post
Uses cached files from src.train / src.predict (no recomputation).
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.config import SEED, WORK_DIR
from src.decision import G, decide, f05_macro, to_lists
from src.features import FEATURES
from src.config import train_paths
from src.io_utils import read_ground_truth
from src.postprocess import VARIANTS, apply_post, numbers_frame


def load_numbers(ndir: Path, split: str):
    s1 = numbers_frame(pl.read_parquet(ndir / f"{split}_s1.parquet", columns=["entity_id", "addr_numbers"]),
                       G, "nums1")
    pool = numbers_frame(pl.concat([pl.read_parquet(ndir / f"{split}_s{k}.parquet",
                                                    columns=["entity_id", "addr_numbers"]) for k in (2, 3)]),
                         "cand_id", "nums2")
    return s1, pool


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--n-valid", type=int, default=100_000)
    args = ap.parse_args()
    mdir, ndir = Path(args.model_dir), Path(args.norm_dir)
    params = json.loads((mdir / "params.json").read_text())["decision"]

    # validation split exactly as in src.train
    s1_ids = pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id"])
    ids = s1_ids.select("entity_id").sample(fraction=1.0, shuffle=True, seed=SEED)["entity_id"]
    n_tr = min(args.n_train, int(len(ids) * 0.8))
    n_va = min(args.n_valid, len(ids) - n_tr)
    train_ids, valid_ids = ids[:n_tr], ids[n_tr:n_tr + n_va]
    feats = pl.read_parquet(mdir / "train_feats.parquet")
    va = feats.filter(~pl.col(G).is_in(train_ids.implode()))
    model = lgb.Booster(model_file=str(mdir / "model.txt"))
    va_scored = va.select(G, "cand_id", "country", "y").with_columns(
        pl.Series("p", model.predict(va.select(FEATURES).to_numpy())))
    gt = read_ground_truth(train_paths(args.data_dir)["gt"])
    truth = gt.filter(pl.col(G).is_in(valid_ids.implode()))   # same truth as src.train
    va_s1n, va_pooln = load_numbers(ndir, "train")

    te_scored = pl.read_parquet(mdir / "test_scored.parquet")
    te_s1 = pl.read_parquet(ndir / "test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": G})
    te_s1n, te_pooln = load_numbers(ndir, "test")

    base_va = decide(va_scored, params).join(va_scored.select(G, "cand_id", "y"), on=[G, "cand_id"])
    base_te = decide(te_scored, params)
    rows = []
    for name in VARIANTS:
        vs = apply_post(va_scored, va_s1n, va_pooln, name)
        sel = decide(vs, params).join(va_scored.select(G, "cand_id", "y"), on=[G, "cand_id"])
        f = f05_macro(to_lists(sel.select(G, "cand_id"), truth[G].to_list()), truth)
        removed = base_va.join(sel, on=[G, "cand_id"], how="anti")
        ts = apply_post(te_scored, te_s1n, te_pooln, name)
        tsel = decide(ts, params)
        per = (te_s1.join(tsel.group_by(G).len(), on=G, how="left").fill_null(0)
               .group_by("country").agg(pl.col("len").mean().round(3)))
        per = {c: v for c, v in per.iter_rows()}
        rows.append({"variant": name, "valid_f05": round(f, 5),
                     "valid_removed_true": int(removed["y"].sum()),
                     "valid_removed_false": int((removed["y"] == 0).sum()),
                     "test_removed": base_te.height - tsel.height,
                     "test_US": per.get("US"), "test_India": per.get("India"),
                     "test_France": per.get("France")})
        print(rows[-1], flush=True)
    with pl.Config(tbl_rows=20, tbl_width_chars=220):
        print(pl.DataFrame(rows))
    print("\nValidation predicted matches per S1 (for comparison):",
          round(base_va.height / truth.height, 3))


if __name__ == "__main__":
    main()
