"""Validation check of the exclusivity-aware ("joint") probability.

The training features file holds scored candidates of 400,000 training S1
records (300k train + 100k validation). Letting ALL of them compete for pool
records, as on test, gives validation S1 entities realistic rivals (with the
caveat that training-split S1 have in-sample, slightly over-confident scores).
Reports validation macro F0.5 (validation S1 only) for highest-probability
exclusivity vs the joint probability with several rival floors.

Usage: python -m src.eval_joint [--model-dir WORK_DIR/model_v3]
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.assign import joint_probability
from src.config import SEED, WORK_DIR, train_paths
from src.decision import G, decide, f05_macro, f05_per_entity, to_lists
from src.features import FEATURES
from src.io_utils import read_ground_truth
from src.postprocess import apply_post, numbers_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model_v3"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--n-valid", type=int, default=100_000)
    args = ap.parse_args()
    mdir, ndir = Path(args.model_dir), Path(args.norm_dir)
    tau = json.loads((mdir / "params.json").read_text())["decision"].get("tau", 0.7)

    ids = (pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id"])
           .sample(fraction=1.0, shuffle=True, seed=SEED)["entity_id"])
    n_tr = min(args.n_train, int(len(ids) * 0.8))
    valid_ids = ids[n_tr:n_tr + min(args.n_valid, len(ids) - n_tr)]
    feats = pl.read_parquet(mdir / "train_feats.parquet")
    model = lgb.Booster(model_file=str(mdir / "model.txt"))
    sc = feats.select(G, "cand_id", "country").with_columns(
        pl.Series("p", model.predict(feats.select(FEATURES).to_numpy())))
    del feats
    s1 = pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id", "addr_numbers"])
    pool = pl.concat([pl.read_parquet(ndir / f"train_s{k}.parquet", columns=["entity_id", "addr_numbers"])
                      for k in (2, 3)])
    sc = apply_post(sc, numbers_frame(s1, G, "nums1"), numbers_frame(pool, "cand_id", "nums2"), "conflict_p90")
    truth = read_ground_truth(train_paths(args.data_dir)["gt"]).filter(pl.col(G).is_in(valid_ids.implode()))
    vids = truth[G].to_list()
    is_val = pl.col(G).is_in(valid_ids.implode())

    def score(scored, label):
        sel = decide(scored, {"rule": "threshold", "tau": tau})
        per = f05_per_entity(to_lists(sel.filter(is_val), vids), truth)
        n = sel.filter(is_val).height
        print(f"{label:38s} valid F0.5 {per['f05'].mean():.5f}  (precision {per['precision'].mean():.4f}, "
              f"recall {per['recall'].mean():.4f}, {n:,} predicted pairs)")

    print(f"\nValidation S1: {len(vids):,}; competing S1: {sc[G].n_unique():,}; tau {tau}")
    score(sc.filter(is_val), "argmax, validation S1 only (as before)")
    score(sc, "argmax, all 400k S1 compete")
    for floor in (0.3, 0.1, 0.05):
        score(joint_probability(sc, None, None, tau, log=lambda *a: None, p_floor=floor),
              f"joint (rival floor {floor}), all compete")


if __name__ == "__main__":
    main()
