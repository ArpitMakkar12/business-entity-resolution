"""Honest validation of the final decision layer of the two-model blend.

Model B (model_v4) was trained on S1 ids[:700k] and validated on ids[700k:800k];
model A (model_v3) was trained on ids[:300k] (early-stopped on ids[300k:400k]).
So ids[700k:800k] are held out for BOTH models, and model B's feature file
already holds their candidates. On these 100,000 entities this script scores
every combination of
  blend weight of model A   w   in {0, 0.3, 0.5, 0.7, 1}
  threshold                 tau in {0.55 .. 0.8}
  post rule                 none / conflict_p90
  joint (exclusivity-aware) probability on / off
and prints the best settings overall and per country.

Usage: python -m src.eval_blend [--model-a WORK_DIR/model_v3] [--model-b WORK_DIR/model_v4]
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.assign import joint_probability
from src.config import SEED, WORK_DIR, train_paths
from src.decision import G, decide, f05_per_entity, to_lists
from src.features import FEATURES
from src.io_utils import read_ground_truth
from src.postprocess import apply_post, numbers_frame
from src.train import log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-a", default=str(Path(WORK_DIR) / "model_v3"))
    ap.add_argument("--model-b", default=str(Path(WORK_DIR) / "model_v4"))
    ap.add_argument("--n-train-b", type=int, default=700_000)
    ap.add_argument("--n-valid-b", type=int, default=100_000)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    ndir, ma, mb = Path(args.norm_dir), Path(args.model_a), Path(args.model_b)

    ids = (pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id"])
           .sample(fraction=1.0, shuffle=True, seed=SEED)["entity_id"])
    n_tr = min(args.n_train_b, int(len(ids) * 0.8))
    valid_ids = ids[n_tr:n_tr + min(args.n_valid_b, len(ids) - n_tr)]
    feats = (pl.scan_parquet(mb / "train_feats.parquet")
             .filter(pl.col(G).is_in(valid_ids.implode())).collect())
    X = feats.select(FEATURES).to_numpy()
    pa = lgb.Booster(model_file=str(ma / "model.txt")).predict(X)
    pb = lgb.Booster(model_file=str(mb / "model.txt")).predict(X)
    base = feats.select(G, "cand_id", "country").with_columns(pl.Series("pa", pa), pl.Series("pb", pb))
    del feats, X
    log(f"held-out S1: {len(valid_ids):,}, candidate pairs {base.height:,}")

    s1 = pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id", "addr_numbers", "country"])
    pool = pl.concat([pl.read_parquet(ndir / f"train_s{k}.parquet", columns=["entity_id", "addr_numbers"])
                      for k in (2, 3)])
    if (mb / "siblings.parquet").exists():
        pool = pl.concat([pool, pl.read_parquet(mb / "siblings.parquet", columns=["entity_id", "addr_numbers"])])
    s1n, pooln = numbers_frame(s1, G, "nums1"), numbers_frame(pool, "cand_id", "nums2")
    truth = read_ground_truth(train_paths(args.data_dir)["gt"]).filter(pl.col(G).is_in(valid_ids.implode()))
    tids = truth[G].to_list()
    country = s1.select(pl.col("entity_id").alias(G), "country")

    rows = []
    for w in (0.0, 0.3, 0.5, 0.7, 1.0):
        sc = base.select(G, "cand_id", "country", (w * pl.col("pa") + (1 - w) * pl.col("pb")).alias("p"))
        for post in ("none", "conflict_p90"):
            sp = sc if post == "none" else apply_post(sc, s1n, pooln, post)
            for joint in (False, True):
                sj = joint_probability(sp, None, None, 0.7, log=lambda *a: None) if joint else sp
                for tau in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
                    per = f05_per_entity(to_lists(decide(sj, {"rule": "threshold", "tau": tau}), tids), truth)
                    per = per.join(country, on=G)
                    by = dict(per.group_by("country").agg(pl.col("f05").mean()).iter_rows())
                    rows.append({"w_A": w, "post": post, "joint": joint, "tau": tau,
                                 "f05": per["f05"].mean(), **{f"f05_{k}": v for k, v in by.items()}})
        log(f"w_A={w} done")
    res = pl.DataFrame(rows).sort("f05", descending=True)
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    print("\n=== top 20 settings (held-out for both models) ===")
    print(res.head(20).with_columns(pl.col(pl.Float64).round(5)))
    print("\n=== reference: current final (w_A 0.5, conflict_p90, joint, tau 0.7) ===")
    print(res.filter((pl.col("w_A") == 0.5) & (pl.col("post") == "conflict_p90") & pl.col("joint")
                     & (pl.col("tau") == 0.7)).with_columns(pl.col(pl.Float64).round(5)))
    print("\n=== w_A 0.5, conflict_p90, joint: by threshold ===")
    print(res.filter((pl.col("w_A") == 0.5) & (pl.col("post") == "conflict_p90") & pl.col("joint"))
          .sort("tau").with_columns(pl.col(pl.Float64).round(5)))


if __name__ == "__main__":
    main()
