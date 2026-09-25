"""Audit test predictions against validation (no test labels needed).

Compares, per country, how many matches the model predicts per S1 entity and
how confident it is, on the validation split (where the truth is known) and on
the test set. A test country that predicts clearly more matches per entity, or
has more mid-confidence matches, is where false merges are likely. Prints
example S1 entities with many predicted matches and example borderline pairs
so the error pattern can be read directly.

Usage:  python -m src.audit [--tau 0.7]
Uses cached files from src.train / src.predict (no recomputation).
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.config import SEED, WORK_DIR
from src.decision import G, decide
from src.features import FEATURES


def per_entity(selected: pl.DataFrame, all_ids: pl.DataFrame) -> pl.DataFrame:
    """(G, country) for every entity + n_pred = number of predicted matches."""
    counts = selected.group_by(G).len().rename({"len": "n_pred"})
    return all_ids.join(counts, on=G, how="left").with_columns(pl.col("n_pred").fill_null(0))


def summary(df: pl.DataFrame, label: str) -> pl.DataFrame:
    return (df.group_by("country").agg(
        pl.len().alias("s1"),
        pl.col("n_pred").mean().round(3).alias("pred_per_s1"),
        (pl.col("n_pred") == 0).mean().round(4).alias("share_0"),
        (pl.col("n_pred") >= 6).mean().round(4).alias("share_6plus"),
    ).with_columns(pl.lit(label).alias("split")).sort("country"))


def band_share(scored: pl.DataFrame, label: str) -> pl.DataFrame:
    """Share of candidate pairs per probability band, per country."""
    labels = ["<.3", ".3-.5", ".5-.7", ".7-.8", ".8-.9", ".9-.97", ">.97"]
    out = (scored.with_columns(pl.col("p").cut([0.3, 0.5, 0.7, 0.8, 0.9, 0.97], labels=labels)
                               .cast(pl.Utf8).alias("band"))
           .group_by("country", "band").len()
           .with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("share"))
           .pivot(on="band", index="country", values="share").sort("country"))
    return out.select(["country"] + [c for c in labels if c in out.columns]).with_columns(
        pl.lit(label).alias("split"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model"))
    ap.add_argument("--n-train", type=int, default=300_000)
    args = ap.parse_args()
    mdir, ndir = Path(args.model_dir), Path(args.norm_dir)
    params = json.loads((mdir / "params.json").read_text())["decision"]
    print("decision rule:", params)

    # ---- validation (truth known): rebuild the exact split used by src.train
    s1_ids = pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id", "country"])
    ids = s1_ids.select("entity_id").sample(fraction=1.0, shuffle=True, seed=SEED)["entity_id"]
    n_tr = min(args.n_train, int(len(ids) * 0.8))
    train_ids = ids[:n_tr]
    feats = pl.read_parquet(mdir / "train_feats.parquet")
    va = feats.filter(~pl.col(G).is_in(train_ids.implode()))
    model = lgb.Booster(model_file=str(mdir / "model.txt"))
    va_scored = va.select(G, "cand_id", "country", "y").with_columns(
        pl.Series("p", model.predict(va.select(FEATURES).to_numpy())))
    va_all = va.select(G, "country").unique()
    va_ent = per_entity(decide(va_scored, params), va_all)
    truth = (va_scored.filter(pl.col("y") == 1).group_by(G).len().rename({"len": "n_true_in_cands"}))
    sel = decide(va_scored, params).join(va_scored.select(G, "cand_id", "y"), on=[G, "cand_id"])
    print(f"\nVALIDATION: precision of predicted pairs = {sel['y'].mean():.4f}, "
          f"true matches in candidates per S1 = {truth['n_true_in_cands'].sum() / va_all.height:.3f}")

    # ---- test
    te_scored = pl.read_parquet(mdir / "test_scored.parquet")
    te_all = pl.read_parquet(ndir / "test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": G})
    te_sel = decide(te_scored, params)
    te_ent = per_entity(te_sel, te_all)

    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        print("\nPredicted matches per S1 (validation vs test):")
        print(pl.concat([summary(va_ent, "valid"), summary(te_ent, "test")]))
        print("\nCandidate probability bands (share of pairs):")
        print(pl.concat([band_share(va_scored, "valid"), band_share(te_scored, "test")], how="diagonal"))
        print("\nDistribution of predicted matches per S1 (test, by country):")
        print(te_ent.with_columns(pl.col("n_pred").clip(upper_bound=10))
              .group_by("country", "n_pred").len().pivot(on="country", index="n_pred", values="len")
              .sort("n_pred"))

    # ---- examples to read
    s1 = pl.read_parquet(ndir / "test_s1.parquet", columns=["entity_id", "business_name", "business_address"])
    pool = pl.concat([pl.read_parquet(ndir / f"test_s{k}.parquet",
                                      columns=["entity_id", "business_name", "business_address"])
                      for k in (2, 3)])
    many = te_ent.filter(pl.col("n_pred") >= 7).sample(n=min(6, te_ent.filter(pl.col("n_pred") >= 7).height),
                                                         seed=SEED)
    ex = (te_sel.filter(pl.col(G).is_in(many[G].implode()))
          .join(te_scored.select(G, "cand_id", "p"), on=[G, "cand_id"])
          .join(pool.rename({"entity_id": "cand_id"}), on="cand_id")
          .join(s1.rename({"entity_id": G, "business_name": "s1_name", "business_address": "s1_addr"}), on=G)
          .sort(G, "p", descending=[False, True]))
    border = (te_sel.join(te_scored.select(G, "cand_id", "p", "country"), on=[G, "cand_id"])
              .filter(pl.col("p") < 0.9).sample(n=25, seed=SEED)
              .join(pool.rename({"entity_id": "cand_id"}), on="cand_id")
              .join(s1.rename({"entity_id": G, "business_name": "s1_name", "business_address": "s1_addr"}), on=G))
    cols = ["country", "s1_name", "business_name", "s1_addr", "business_address", "p"]
    with pl.Config(tbl_rows=80, fmt_str_lengths=45, tbl_width_chars=260):
        print("\nTest S1 entities with 7+ predicted matches (S1 vs each predicted record):")
        print(ex.join(te_all, on=G).select([G] + cols))
        print("\nRandom predicted test pairs with 0.7 <= p < 0.9 (borderline):")
        print(border.select(cols))


if __name__ == "__main__":
    main()
