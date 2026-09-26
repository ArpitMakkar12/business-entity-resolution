"""Where does validation F0.5 go? Loss attribution + examples.

Re-scores the cached validation pairs (the exact split of src.train), applies
the tuned decision rule, and splits the lost F0.5 (1 - score) per entity into:

  singleton_fp      singleton entity that received >= 1 prediction
  missed_all        entity with matches that received no prediction
  mixed             entity with both false merges and missed matches
  false_merge_only  all true matches found, but extra wrong ones added
  missed_some       only missed matches (no false merges); split further into
                    matches outside the candidate set (blocking) vs matches
                    the model rejected

Loss is reported per country and as a share of the total loss, followed by
example rows for each category so the pattern can be read directly.

Usage:  python -m src.error_analysis
Uses cached files from src.train (no recomputation).
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.config import SEED, WORK_DIR, train_paths
from src.decision import G, decide, f05_per_entity, to_lists
from src.features import FEATURES
from src.io_utils import explode_ground_truth, read_ground_truth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--n-valid", type=int, default=100_000)
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()
    mdir, ndir = Path(args.model_dir), Path(args.norm_dir)
    params = json.loads((mdir / "params.json").read_text())["decision"]

    # exact validation split of src.train
    s1 = pl.read_parquet(ndir / "train_s1.parquet",
                         columns=["entity_id", "business_name", "business_address", "country"])
    ids = s1.select("entity_id").sample(fraction=1.0, shuffle=True, seed=SEED)["entity_id"]
    n_tr = min(args.n_train, int(len(ids) * 0.8))
    n_va = min(args.n_valid, len(ids) - n_tr)
    valid_ids = ids[n_tr:n_tr + n_va]
    gt = read_ground_truth(train_paths(args.data_dir)["gt"])
    truth = gt.filter(pl.col(G).is_in(valid_ids.implode()))
    pairs_true = explode_ground_truth(truth)

    feats = pl.read_parquet(mdir / "train_feats.parquet").filter(pl.col(G).is_in(valid_ids.implode()))
    model = lgb.Booster(model_file=str(mdir / "model.txt"))
    scored = feats.select(G, "cand_id", "y").with_columns(
        pl.Series("p", model.predict(feats.select(FEATURES).to_numpy())))
    chosen = decide(scored, params)
    per = f05_per_entity(to_lists(chosen, truth[G].to_list()), truth)
    per = per.join(s1.select(pl.col("entity_id").alias(G), "country"), on=G)

    cand_set = scored.select(G, pl.col("cand_id").alias("matched_id"))
    blocked_miss = (pairs_true.join(cand_set, on=[G, "matched_id"], how="anti")
                    .group_by(G).len().rename({"len": "n_block_miss"}))
    per = (per.with_columns(
        pl.col("ids").list.set_difference(pl.col("matched")).list.len().alias("n_fp"),
        pl.col("matched").list.set_difference(pl.col("ids")).list.len().alias("n_fn"),
        pl.col("matched").list.len().alias("n_true"),
        (1.0 - pl.col("f05")).alias("loss"))
        .join(blocked_miss, on=G, how="left").with_columns(pl.col("n_block_miss").fill_null(0)))
    cat = (pl.when(pl.col("is_singleton") & (pl.col("n_fp") > 0)).then(pl.lit("singleton_fp"))
           .when(~pl.col("is_singleton") & (pl.col("ids").list.len() == 0)).then(pl.lit("missed_all"))
           .when((pl.col("n_fp") > 0) & (pl.col("n_fn") > 0)).then(pl.lit("mixed"))
           .when(pl.col("n_fp") > 0).then(pl.lit("false_merge_only"))
           .when(pl.col("n_fn") > 0).then(pl.lit("missed_some"))
           .otherwise(pl.lit("perfect")))
    per = per.with_columns(cat.alias("category"))
    total_loss = per["loss"].sum()
    n = per.height
    print(f"validation entities: {n:,}   macro F0.5 = {per['f05'].mean():.5f}   "
          f"(total loss {total_loss / n:.5f} per entity)\n")

    summary = (per.group_by("category").agg(
        pl.len().alias("entities"),
        (pl.col("loss").sum() / n).round(5).alias("F05_points_lost"),
        (pl.col("loss").sum() / total_loss).round(3).alias("share_of_loss"),
        pl.col("n_fp").sum().alias("false_pairs"),
        pl.col("n_fn").sum().alias("missed_pairs"),
        pl.col("n_block_miss").sum().alias("missed_by_blocking"))
        .sort("F05_points_lost", descending=True))
    by_country = (per.group_by("country", "category").agg((pl.col("loss").sum()).alias("l"))
                  .join(per.group_by("country").len(), on="country")
                  .with_columns((pl.col("l") / pl.col("len")).round(5).alias("points_lost"))
                  .pivot(on="category", index="country", values="points_lost").fill_null(0.0))
    fn_total = per["n_fn"].sum()
    blk = per["n_block_miss"].sum()
    fp_pairs = chosen.join(scored.select(G, "cand_id", "y"), on=[G, "cand_id"]).filter(pl.col("y") == 0)
    with pl.Config(tbl_rows=20, tbl_width_chars=220):
        print(summary)
        print("\nF0.5 points lost per country and category:")
        print(by_country)
    print(f"\nmissed true pairs: {fn_total:,}  of which outside the candidate set (blocking): "
          f"{blk:,} ({blk / max(fn_total, 1):.1%}), rejected by the model: {fn_total - blk:,}")
    print(f"false predicted pairs: {fp_pairs.height:,}  of which synthetic siblings: "
          f"{fp_pairs.filter(pl.col('cand_id').str.contains('-syn')).height:,}")

    # ---------------- examples
    pool = pl.concat([pl.read_parquet(ndir / f"train_s{k}.parquet",
                                      columns=["entity_id", "business_name", "business_address"])
                      for k in (2, 3)])
    sib_path = mdir / "siblings.parquet"
    if sib_path.exists():
        pool = pl.concat([pool, pl.read_parquet(sib_path, columns=pool.columns)])
    s1v = s1.rename({"entity_id": G, "business_name": "s1_name", "business_address": "s1_addr"})
    ps = scored.select(G, "cand_id", "p")

    def show(title, frame):
        if frame.height == 0:
            return
        ex = (frame.sample(n=min(args.examples, frame.height), seed=SEED)
              .join(s1v, on=G).join(pool.rename({"entity_id": "cand_id"}), on="cand_id", how="left")
              .join(ps, on=[G, "cand_id"], how="left"))
        with pl.Config(tbl_rows=40, fmt_str_lengths=42, tbl_width_chars=260):
            print(f"\n{title}")
            print(ex.select("country", "s1_name", "business_name", "s1_addr", "business_address", "p"))

    fp_real = fp_pairs.filter(~pl.col("cand_id").str.contains("-syn")).select(G, "cand_id")
    show("FALSE MERGES (predicted, not in truth; real records):", fp_real.join(per.select(G, "country"), on=G))
    fn_model = (pairs_true.join(chosen.rename({"cand_id": "matched_id"}), on=[G, "matched_id"], how="anti")
                .join(cand_set, on=[G, "matched_id"], how="semi").rename({"matched_id": "cand_id"}))
    show("MISSED - in candidates but rejected by the model:", fn_model.join(per.select(G, "country"), on=G))
    fn_block = pairs_true.join(cand_set, on=[G, "matched_id"], how="anti").rename({"matched_id": "cand_id"})
    show("MISSED - never reached the candidate set (blocking):", fn_block.join(per.select(G, "country"), on=G))


if __name__ == "__main__":
    main()
