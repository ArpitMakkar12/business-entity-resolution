"""Diagnostic: look-alike ("sibling") entities INSIDE Source 1, train vs test.

Leaderboard probes showed that test has far more S1 entities without any true
match than train (5.6%). One explanation: test S1 contains sibling businesses of
other S1 entities (same core name, neighbouring house number) that have no copies
of their own, so they compete with the real entity for its copies.

For train and test S1 it reports, per country:
  same_name   share of S1 whose core name (legal form removed) is shared by another S1
  near        ... and whose first house numbers differ by 1..30 (sibling pattern)
  dup         ... with the same first house number
Train: the true no-match rate inside each group (from the ground truth).
Test:  our predicted no-match rate inside each group, and for sibling pairs the
       share where BOTH members received matches.
Also compares the distribution of true match counts (train) with predicted
counts (test).

Usage: python -m src.diag_s1 --pred <dir with matching_results.tsv>
"""
import argparse
from pathlib import Path

import polars as pl

from src.config import WORK_DIR, train_paths
from src.io_utils import read_ground_truth

G = "source1_entity_id"


def groups(ndir: Path, split: str) -> pl.DataFrame:
    s1 = pl.read_parquet(ndir / f"{split}_s1.parquet", columns=["entity_id", "country", "name_core",
                                                                  "addr_numbers"])
    d = s1.select(pl.col("entity_id").alias(G), "country", "name_core",
                  pl.col("addr_numbers").list.first().cast(pl.Int64, strict=False).alias("n1"))
    ok = d.filter(pl.col("name_core").str.len_chars() >= 3)
    sz = ok.group_by("country", "name_core").len()
    ok = ok.join(sz.filter(pl.col("len").is_between(2, 50)), on=["country", "name_core"])
    j = (ok.join(ok, on=["country", "name_core"], suffix="_b")
         .filter(pl.col(G) != pl.col(f"{G}_b"))
         .with_columns((pl.col("n1") - pl.col("n1_b")).abs().alias("dn")))
    flags = j.group_by(G).agg(
        pl.lit(True).alias("same_name"),
        pl.col("dn").is_between(1, 30).any().alias("near"),
        (pl.col("dn") == 0).any().alias("dup"))
    out = d.select(G, "country").join(flags, on=G, how="left").fill_null(False)
    return out, j.filter(pl.col("dn").is_between(1, 30)).select(G, f"{G}_b", "country")


def read_pred(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    col = df.columns[1]
    return df.select(pl.col(G), pl.col(col).fill_null("").str.split(",")
                     .list.eval(pl.element().filter(pl.element() != "")).list.len().alias("n_pred"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--pred", default=str(Path(WORK_DIR) / "output_ens_p90"))
    args = ap.parse_args()
    ndir = Path(args.norm_dir)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)

    tr, _ = groups(ndir, "train")
    gt = read_ground_truth(train_paths(args.data_dir)["gt"]).select(G, "n_matches")
    tr = tr.join(gt, on=G, how="left").with_columns(pl.col("n_matches").fill_null(0))
    te, te_pairs = groups(ndir, "test")
    te = te.join(read_pred(Path(args.pred) / "matching_results.tsv"), on=G, how="left")

    print("\n=== TRAIN: share of S1 in each group, and TRUE no-match rate inside it ===")
    print(tr.group_by("country").agg(
        pl.len().alias("n_s1"),
        pl.col("same_name").mean().round(4), pl.col("near").mean().round(4), pl.col("dup").mean().round(4),
        (pl.col("n_matches") == 0).mean().round(4).alias("nomatch_all"),
        (pl.col("n_matches") == 0).filter(pl.col("same_name")).mean().round(4).alias("nomatch_same"),
        (pl.col("n_matches") == 0).filter(pl.col("near")).mean().round(4).alias("nomatch_near"),
        (pl.col("n_matches") == 0).filter(pl.col("dup")).mean().round(4).alias("nomatch_dup"),
    ).sort("country"))

    print("\n=== TEST: share of S1 in each group, and PREDICTED no-match rate inside it ===")
    print(te.group_by("country").agg(
        pl.len().alias("n_s1"),
        pl.col("same_name").mean().round(4), pl.col("near").mean().round(4), pl.col("dup").mean().round(4),
        (pl.col("n_pred") == 0).mean().round(4).alias("pred_empty_all"),
        (pl.col("n_pred") == 0).filter(pl.col("same_name")).mean().round(4).alias("pred_empty_same"),
        (pl.col("n_pred") == 0).filter(pl.col("near")).mean().round(4).alias("pred_empty_near"),
        (pl.col("n_pred") == 0).filter(pl.col("dup")).mean().round(4).alias("pred_empty_dup"),
    ).sort("country"))

    both = (te_pairs.join(te.select(G, "n_pred"), on=G)
            .join(te.select(pl.col(G).alias(f"{G}_b"), pl.col("n_pred").alias("n_pred_b")), on=f"{G}_b"))
    print("\n=== TEST sibling pairs (same core name, numbers 1..30 apart) ===")
    print(both.group_by("country").agg(
        pl.len().alias("pairs"),
        ((pl.col("n_pred") > 0) & (pl.col("n_pred_b") > 0)).mean().round(4).alias("both_matched"),
        ((pl.col("n_pred") > 0) ^ (pl.col("n_pred_b") > 0)).mean().round(4).alias("one_matched"),
        ((pl.col("n_pred") == 0) & (pl.col("n_pred_b") == 0)).mean().round(4).alias("none_matched"),
    ).sort("country"))

    print("\n=== Match-count distribution: TRAIN truth vs TEST prediction (share of S1) ===")
    def dist(df, col, name):
        return (df.with_columns(pl.col(col).clip(upper_bound=6).alias("k"))
                .group_by("country", "k").len()
                .with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias(name))
                .drop("len"))
    print(dist(tr, "n_matches", "train_true").join(dist(te, "n_pred", "test_pred"), on=["country", "k"],
                                                   how="full", coalesce=True).sort("country", "k"))


if __name__ == "__main__":
    main()
