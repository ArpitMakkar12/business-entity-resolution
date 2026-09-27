"""Name tokens that are much more frequent on test than on train (new noise types).

The probability bands show that true test matches score lower than validation
matches (test copies are noisier), and the unclaimed records in the 0.3-0.7
band are often real copies with an added word ("Shri Gem Life Private Ltd" vs
"Gem Life Private Limited", "NB Silk Limited Center"). A noise type that exists
only (or mostly) in test was never learned as harmless, so every copy with it
loses probability.

For one country this compares the normalized name tokens of the S2/S3 pool
(and of S1) between train and test: rate per 100,000 records as first token,
last token and anywhere, sorted by the test excess.

Usage: python -m src.diag_tokens --country India
"""
import argparse
from pathlib import Path

import polars as pl

from src.config import WORK_DIR


def rates(df: pl.DataFrame, where: str) -> pl.DataFrame:
    n = df.height
    if where == "first":
        t = df.select(pl.col("name_tokens").list.first().alias("tok"))
    elif where == "last":
        t = df.select(pl.col("name_tokens").list.last().alias("tok"))
    else:
        t = df.select(pl.col("name_tokens").list.unique().alias("tok")).explode("tok")
    return (t.drop_nulls().group_by("tok").len()
            .with_columns((pl.col("len") * 100_000 / n).alias("per_100k")).drop("len"))


def compare(tr: pl.DataFrame, te: pl.DataFrame, label: str, top: int):
    for where in ("first", "last", "any"):
        a = rates(tr, where).rename({"per_100k": "train"})
        b = rates(te, where).rename({"per_100k": "test"})
        m = (b.join(a, on="tok", how="left").fill_null(0.0)
             .with_columns((pl.col("test") - pl.col("train")).alias("excess"),
                           (pl.col("test") / (pl.col("train") + 5)).alias("ratio"))
             .filter(pl.col("test") >= 50))
        print(f"\n=== {label}: token as {where.upper()} word, per 100k records (largest test excess) ===")
        print(m.sort("excess", descending=True).head(top).with_columns(pl.col(pl.Float64).round(1)))
        print(f"--- {label}: {where} tokens with the highest test/train ratio (test >= 200 per 100k) ---")
        print(m.filter(pl.col("test") >= 200).sort("ratio", descending=True).head(12)
              .with_columns(pl.col(pl.Float64).round(1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--country", default="India")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    ndir, c = Path(args.norm_dir), args.country
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)

    def load(split, srcs):
        return pl.concat([pl.read_parquet(ndir / f"{split}_s{k}.parquet", columns=["name_tokens", "country"])
                          for k in srcs]).filter(pl.col("country") == c)
    tr_pool, te_pool = load("train", (2, 3)), load("test", (2, 3))
    if tr_pool.height == 0:
        print(f"no training records for {c}; comparing with all training countries")
        tr_pool = pl.concat([pl.read_parquet(ndir / f"train_s{k}.parquet", columns=["name_tokens", "country"])
                             for k in (2, 3)])
    compare(tr_pool, te_pool, f"{c} S2/S3 pool", args.top)
    tr_s1, te_s1 = load("train", (1,)), load("test", (1,))
    if tr_s1.height:
        a = rates(tr_s1, "first").rename({"per_100k": "train"})
        b = rates(te_s1, "first").rename({"per_100k": "test"})
        print(f"\n=== {c} S1 (clean names): first word, largest test excess ===")
        print(b.join(a, on="tok", how="left").fill_null(0.0)
              .with_columns((pl.col("test") - pl.col("train")).alias("excess"))
              .sort("excess", descending=True).head(15).with_columns(pl.col(pl.Float64).round(1)))


if __name__ == "__main__":
    main()
