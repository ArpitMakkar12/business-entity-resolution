"""Are we missing true matches on test? (calibration bands + unclaimed pool records)

The adversarial check showed that India test PREDICTIONS look like validation
predictions. The remaining question is what we do NOT predict:

  A  probability bands per S1 entity (after the p90 rule and exclusivity),
     validation vs test: if test has more pairs in the uncertain band
     (0.3-0.7) and fewer in the confident band, true test matches are scored
     lower than validation matches (noisier copies) and a lower threshold pays.
     Validation precision per band shows what each band is worth.
  B  pool density: train pool records per S1 and true matches per S1 (ground
     truth) vs test pool records per S1 and predicted matches per S1.
  C  unclaimed test pool records (not predicted for any S1): share never in a
     candidate list, distribution of their best probability, and examples with
     their best S1 candidate, to eyeball whether they are missed copies or
     distractor businesses.

Usage: python -m src.diag_unclaimed --country India [--model-dir WORK_DIR/model_v3]
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.config import SEED, WORK_DIR, train_paths
from src.decision import G, apply_exclusivity, decide
from src.features import FEATURES
from src.io_utils import explode_ground_truth, read_ground_truth
from src.postprocess import apply_post, numbers_frame
from src.train import log

BANDS = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.99), (0.99, 1.01)]


def bands(ex: pl.DataFrame, n_s1: int, label: bool) -> pl.DataFrame:
    rows = []
    for lo, hi in BANDS:
        b = ex.filter(pl.col("p").is_between(lo, hi, closed="left"))
        rows.append((f"{lo:.2f}-{min(hi, 1):.2f}", b.height / n_s1,
                     float(b["y"].mean()) if label and b.height else None))
    return pl.DataFrame(rows, schema=["band", "pairs_per_s1", "precision"], orient="row")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model_v3"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--country", default="India")
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--n-valid", type=int, default=100_000)
    args = ap.parse_args()
    mdir, ndir, c = Path(args.model_dir), Path(args.norm_dir), args.country
    tau = json.loads((mdir / "params.json").read_text())["decision"].get("tau", 0.7)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_fmt_str_lengths(40)

    # ---------------- A: probability bands --------------------------------------
    ids = (pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id"])
           .sample(fraction=1.0, shuffle=True, seed=SEED)["entity_id"])
    n_tr = min(args.n_train, int(len(ids) * 0.8))
    valid_ids = ids[n_tr:n_tr + min(args.n_valid, len(ids) - n_tr)]
    vf = (pl.scan_parquet(mdir / "train_feats.parquet")
          .filter(pl.col(G).is_in(valid_ids.implode()) & (pl.col("country") == c)).collect())
    model = lgb.Booster(model_file=str(mdir / "model.txt"))
    vf = vf.select(G, "cand_id", "country", "y").with_columns(
        pl.Series("p", model.predict(vf.select(FEATURES).to_numpy())))
    tr_s1 = pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id", "addr_numbers", "country"])
    tr_pool = pl.concat([pl.read_parquet(ndir / f"train_s{k}.parquet", columns=["entity_id", "addr_numbers",
                                                                                 "country"]) for k in (2, 3)])
    vs = apply_post(vf.select(G, "cand_id", "country", "p"), numbers_frame(tr_s1, G, "nums1"),
                    numbers_frame(tr_pool, "cand_id", "nums2"), "conflict_p90")
    vex = apply_exclusivity(vs).join(vf.select(G, "cand_id", "y"), on=[G, "cand_id"])
    n_val = vf[G].n_unique()

    s1 = pl.read_parquet(ndir / "test_s1.parquet", columns=["entity_id", "business_name", "business_address",
                                                             "addr_numbers", "country"])
    pool = pl.concat([pl.read_parquet(ndir / f"test_s{k}.parquet", columns=[
        "entity_id", "business_name", "business_address", "addr_numbers", "country"]) for k in (2, 3)])
    sc = pl.read_parquet(mdir / "test_scored.parquet")
    sc = apply_post(sc, numbers_frame(s1, G, "nums1"), numbers_frame(pool, "cand_id", "nums2"), "conflict_p90")
    sc_c = sc.filter(pl.col("country") == c)
    tex = apply_exclusivity(sc_c)
    n_te = s1.filter(pl.col("country") == c).height
    bv, bt = bands(vex, n_val, True), bands(tex, n_te, False)
    print(f"\n=== A. {c}: pairs per S1 entity by probability band (after p90 + exclusivity) ===")
    print(bv.rename({"pairs_per_s1": "valid_per_s1", "precision": "valid_precision"})
          .join(bt.select("band", pl.col("pairs_per_s1").alias("test_per_s1")), on="band")
          .with_columns(pl.col(pl.Float64).round(4)))

    # ---------------- B: pool density --------------------------------------------
    gt = explode_ground_truth(read_ground_truth(train_paths(args.data_dir)["gt"]))
    tr_s1c = tr_s1.filter(pl.col("country") == c)
    tr_poolc = tr_pool.filter(pl.col("country") == c)
    true_c = gt.join(tr_s1c.select(pl.col("entity_id").alias(G)), on=G).height
    sel = decide(sc_c, {"rule": "threshold", "tau": tau})
    poolc = pool.filter(pl.col("country") == c)
    print(f"\n=== B. {c}: pool density ===")
    print(f"  train: {tr_s1c.height:,} S1, {tr_poolc.height:,} pool records = {tr_poolc.height / tr_s1c.height:.2f}"
          f"/S1, true matches {true_c / tr_s1c.height:.2f}/S1 ({true_c / tr_poolc.height:.1%} of pool records)")
    print(f"  test:  {n_te:,} S1, {poolc.height:,} pool records = {poolc.height / n_te:.2f}/S1, "
          f"PREDICTED matches {sel.height / n_te:.2f}/S1 ({sel.height / poolc.height:.1%} of pool records)")
    print(f"  validation: predicted {vex.filter(pl.col('p') >= tau).height / n_val:.2f}/S1, "
          f"precision {vex.filter(pl.col('p') >= tau)['y'].mean():.4f}")

    # ---------------- C: unclaimed pool records -----------------------------------
    cands = pl.read_parquet(mdir / "test_cands.parquet", columns=[G, "cand_id"])
    in_cand = poolc.select(pl.col("entity_id").alias("cand_id")).join(cands.select("cand_id").unique(),
                                                                      on="cand_id", how="semi")
    unclaimed = poolc.select(pl.col("entity_id").alias("cand_id")).join(sel.select("cand_id"), on="cand_id",
                                                                        how="anti")
    best = (sc_c.join(unclaimed, on="cand_id").sort("p", descending=True)
            .group_by("cand_id", maintain_order=True).head(1))
    print(f"\n=== C. {c}: unclaimed test pool records ===")
    print(f"  pool records {poolc.height:,}; in some candidate list {in_cand.height / poolc.height:.1%}; "
          f"unclaimed {unclaimed.height:,} ({unclaimed.height / poolc.height:.1%}); "
          f"unclaimed with no candidate S1 at all: {unclaimed.height - best.height:,}")
    print("  best probability of unclaimed records: " + ", ".join(
        f"{lo:.1f}-{hi:.1f}: {best.filter(pl.col('p').is_between(lo, hi, closed='left')).height:,}"
        for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01))))
    a = s1.select(pl.col("entity_id").alias(G), pl.col("business_name").alias("best_s1_name"),
                  pl.col("business_address").alias("best_s1_addr"))
    b = pool.select(pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("rec_name"),
                    pl.col("business_address").alias("rec_addr"))
    for lo, hi, title in ((0.3, 0.7, "best p 0.3-0.7"), (0.05, 0.3, "best p 0.05-0.3"), (0.0, 0.05, "best p < 0.05")):
        ex = best.filter(pl.col("p").is_between(lo, hi, closed="left"))
        print(f"\n  --- unclaimed records, {title} (random 20 of {ex.height:,}) ---")
        print(ex.sample(n=min(20, ex.height), seed=7).join(b, on="cand_id").join(a, on=G)
              .select("rec_name", "rec_addr", "best_s1_name", "best_s1_addr", pl.col("p").round(3)))
    nc = unclaimed.join(best.select("cand_id"), on="cand_id", how="anti")
    if nc.height:
        print(f"\n  --- unclaimed records never in any candidate list (random 15 of {nc.height:,}) ---")
        print(nc.sample(n=min(15, nc.height), seed=8).join(b, on="cand_id").select("rec_name", "rec_addr"))


if __name__ == "__main__":
    main()
