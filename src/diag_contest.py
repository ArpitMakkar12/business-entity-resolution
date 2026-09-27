"""Diagnostic: S2/S3 records claimed by several S1 entities on test.

Validation cannot show competition between S1 entities (only 100k of 2.2M
training S1 records are scored), but on test every S1 record is scored, so a
pool record can be likely (p >= tau) for two S1 entities at once, e.g. two
look-alike or duplicate S1 records. Exclusivity then gives each record to the
S1 with the higher probability, which can SPLIT one business's copies between
two S1 entities (both then score badly).

Reports per country: share of S1 entities involved in contested records, and
how many contested S1 pairs end up split (both keep some shared records).

Usage: python -m src.diag_contest [--model-dir WORK_DIR/model_v3] [--blend DIR]
"""
import argparse
from pathlib import Path

import polars as pl

from src.config import WORK_DIR
from src.decision import G
from src.postprocess import apply_post, numbers_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model_v3"))
    ap.add_argument("--blend", nargs="*", default=[])
    ap.add_argument("--tau", type=float, default=0.7)
    args = ap.parse_args()
    ndir, mdir = Path(args.norm_dir), Path(args.model_dir)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)

    sc = pl.read_parquet(mdir / "test_scored.parquet")
    cols = ["p"]
    for i, d in enumerate(args.blend, 1):
        sc = sc.join(pl.read_parquet(Path(d) / "test_scored.parquet").select(G, "cand_id", pl.col("p").alias(f"p{i}")),
                     on=[G, "cand_id"], how="left")
        cols.append(f"p{i}")
    sc = sc.with_columns(pl.mean_horizontal(cols).alias("p")).drop(cols[1:])
    s1 = pl.read_parquet(ndir / "test_s1.parquet", columns=["entity_id", "addr_numbers", "country"])
    pool = pl.concat([pl.read_parquet(ndir / f"test_s{k}.parquet", columns=["entity_id", "addr_numbers"])
                      for k in (2, 3)])
    sc = apply_post(sc, numbers_frame(s1, G, "nums1"), numbers_frame(pool, "cand_id", "nums2"), "conflict_p90")
    hi = sc.filter(pl.col("p") >= args.tau).select(G, "cand_id", "country", "p")
    hi = hi.with_columns(pl.len().over("cand_id").alias("n_claims"),
                         (pl.col("p") == pl.col("p").max().over("cand_id")).alias("wins"))
    n_s1 = s1.group_by("country").len().rename({"len": "n_s1"})

    per_s1 = hi.group_by(G, "country").agg(
        (pl.col("n_claims") >= 2).any().alias("contested"),
        (pl.col("wins") & (pl.col("n_claims") >= 2)).sum().alias("won"),
        (~pl.col("wins") & (pl.col("n_claims") >= 2)).sum().alias("lost"),
        pl.col("wins").sum().alias("kept"))
    print("\n=== Test S1 entities involved in contested records (p >= tau for 2+ S1) ===")
    print(per_s1.group_by("country").agg(
        pl.col("contested").sum().alias("s1_contested"),
        (pl.col("contested") & (pl.col("won") > 0) & (pl.col("lost") > 0)).sum().alias("s1_won_and_lost"),
        (pl.col("contested") & (pl.col("kept") == 0)).sum().alias("s1_lost_everything"),
    ).join(n_s1, on="country").with_columns(
        (pl.col("s1_contested") / pl.col("n_s1")).round(4).alias("share_contested"),
        (pl.col("s1_won_and_lost") / pl.col("n_s1")).round(4).alias("share_split"),
    ).sort("country"))

    c = hi.filter(pl.col("n_claims") >= 2)
    pairs = (c.join(c.select("cand_id", pl.col(G).alias("other"), pl.col("wins").alias("o_wins")), on="cand_id")
             .filter(pl.col(G) < pl.col("other"))
             .group_by(G, "other", "country").agg(pl.len().alias("shared"),
                                                  pl.col("wins").sum().alias("a_wins"),
                                                  pl.col("o_wins").sum().alias("b_wins")))
    print("\n=== Contested S1 pairs: are the shared records split between the two? ===")
    print(pairs.group_by("country").agg(
        pl.len().alias("pairs"),
        (pl.col("shared") >= 2).mean().round(4).alias("share_2plus_shared"),
        ((pl.col("a_wins") > 0) & (pl.col("b_wins") > 0)).mean().round(4).alias("split"),
        pl.col("shared").mean().round(2).alias("mean_shared"),
    ).sort("country"))
    print("\nExamples of split pairs:")
    ex = pairs.filter((pl.col("a_wins") > 0) & (pl.col("b_wins") > 0)).head(15)
    names = pl.read_parquet(ndir / "test_s1.parquet", columns=["entity_id", "business_name", "business_address"])
    print(ex.join(names.rename({"entity_id": G, "business_name": "name_a", "business_address": "addr_a"}), on=G)
          .join(names.rename({"entity_id": "other", "business_name": "name_b", "business_address": "addr_b"}),
                on="other")
          .select("country", "name_a", "addr_a", "name_b", "addr_b", "shared", "a_wins", "b_wins"))


if __name__ == "__main__":
    main()
