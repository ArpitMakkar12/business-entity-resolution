"""Diagnostic: which kinds of predicted entities are over-represented on test?

Every S1 entity is put in a cell by what we predict for it (after exclusivity,
the p90 conflict rule and the decision threshold):
  k      number of predicted matches (0, 1, 2, 3+)
  pmax   highest probability among them (<.95, .95-.99, >=.99)
  src    predicted matches come from one source only (S2 or S3) or from both
  num    at least one predicted match shares a house number with the S1 (or
         the S1 has no number)
On VALIDATION (labels known) it reports each cell's share of entities, the
share of them that truly have no match, and their mean F0.5. On TEST it reports
each cell's share. Cells much more frequent on test than on validation are
where test-only errors (entities without a real match that still get one)
concentrate; ``gain_if_blank`` estimates what predicting "no match" for the
whole cell would change on test, assuming the excess entities have no match.

Usage: python -m src.diag_k [--model-dir WORK_DIR/model_v3]
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.config import SEED, WORK_DIR, train_paths
from src.decision import G, decide, f05_per_entity, to_lists
from src.features import FEATURES
from src.io_utils import read_ground_truth
from src.postprocess import apply_post, numbers_frame


def load_numbers(ndir: Path, split: str):
    s1 = pl.read_parquet(ndir / f"{split}_s1.parquet", columns=["entity_id", "addr_numbers", "country"])
    pool = pl.concat([pl.read_parquet(ndir / f"{split}_s{k}.parquet", columns=["entity_id", "addr_numbers"])
                      for k in (2, 3)])
    return s1, numbers_frame(s1, G, "nums1"), numbers_frame(pool, "cand_id", "nums2")


def cells(sel: pl.DataFrame, s1: pl.DataFrame, s1n, pooln) -> pl.DataFrame:
    """Per S1 entity: k, pmax, one/both sources, number agreement -> cell label."""
    d = (sel.join(s1n, on=G, how="left").join(pooln, on="cand_id", how="left")
         .with_columns((pl.col("nums1").list.set_intersection(pl.col("nums2")).list.len() > 0)
                       .fill_null(False).alias("nm"),
                       (pl.col("nums1").list.len().fill_null(0) == 0).alias("s1_nonum")))
    per = d.group_by(G).agg(pl.len().alias("k"), pl.col("p").max().alias("pmax"),
                            pl.col("cand_id").str.slice(0, 2).n_unique().alias("nsrc"),
                            (pl.col("nm").any() | pl.col("s1_nonum").first()).alias("num_ok"))
    out = s1.select(pl.col("entity_id").alias(G), "country").join(per, on=G, how="left")
    return out.with_columns(
        pl.col("k").fill_null(0),
        pl.when(pl.col("k").fill_null(0) == 0).then(pl.lit("k0"))
        .otherwise(pl.concat_str([
            pl.when(pl.col("k") >= 3).then(pl.lit("k3+")).otherwise(pl.lit("k") + pl.col("k").cast(pl.Utf8)),
            pl.when(pl.col("pmax") >= 0.99).then(pl.lit("p>=.99"))
            .when(pl.col("pmax") >= 0.95).then(pl.lit("p.95-.99")).otherwise(pl.lit("p<.95")),
            pl.when(pl.col("nsrc") >= 2).then(pl.lit("2src")).otherwise(pl.lit("1src")),
            pl.when(pl.col("num_ok")).then(pl.lit("num_ok")).otherwise(pl.lit("num_diff"))],
            separator=" ")).alias("cell"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model_v3"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--n-valid", type=int, default=100_000)
    ap.add_argument("--post", default="conflict_p90")
    args = ap.parse_args()
    mdir, ndir = Path(args.model_dir), Path(args.norm_dir)
    params = json.loads((mdir / "params.json").read_text())["decision"]
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_width_chars(220)

    ids = (pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id"])
           .sample(fraction=1.0, shuffle=True, seed=SEED)["entity_id"])
    n_tr = min(args.n_train, int(len(ids) * 0.8))
    valid_ids = ids[n_tr:n_tr + min(args.n_valid, len(ids) - n_tr)]
    feats = pl.read_parquet(mdir / "train_feats.parquet").filter(pl.col(G).is_in(valid_ids.implode()))
    model = lgb.Booster(model_file=str(mdir / "model.txt"))
    va = feats.select(G, "cand_id", "country").with_columns(
        pl.Series("p", model.predict(feats.select(FEATURES).to_numpy())))
    del feats
    tr_s1, tr_s1n, tr_pooln = load_numbers(ndir, "train")
    va = apply_post(va, tr_s1n, tr_pooln, args.post)
    va_sel = decide(va, params).join(va.select(G, "cand_id", "p"), on=[G, "cand_id"])
    truth = read_ground_truth(train_paths(args.data_dir)["gt"]).filter(pl.col(G).is_in(valid_ids.implode()))
    va_c = cells(va_sel, tr_s1.filter(pl.col("entity_id").is_in(valid_ids.implode())), tr_s1n, tr_pooln)
    per = f05_per_entity(to_lists(va_sel.select(G, "cand_id"), truth[G].to_list()), truth)
    va_c = va_c.join(per.select(G, "f05", "is_singleton"), on=G)

    te = pl.read_parquet(mdir / "test_scored.parquet")
    te_s1, te_s1n, te_pooln = load_numbers(ndir, "test")
    te = apply_post(te, te_s1n, te_pooln, args.post)
    te_sel = decide(te, params).join(te.select(G, "cand_id", "p"), on=[G, "cand_id"])
    te_c = cells(te_sel, te_s1, te_s1n, te_pooln)

    for country in ("India", "US", "France"):
        v = va_c if country == "France" else va_c.filter(pl.col("country") == country)
        t = te_c.filter(pl.col("country") == country)
        vs = v.group_by("cell").agg((pl.len() / v.height).alias("val_share"),
                                    pl.col("is_singleton").mean().alias("val_nomatch"),
                                    pl.col("f05").mean().alias("val_f05"))
        ts = t.group_by("cell").agg((pl.len() / t.height).alias("test_share"))
        m = (ts.join(vs, on="cell", how="left").fill_null(0.0)
             .with_columns((1 - pl.col("val_share") / pl.col("test_share")).clip(lower_bound=0).alias("x"))
             .with_columns((pl.col("test_share") * (pl.col("x") + (1 - pl.col("x"))
                                                     * (pl.col("val_nomatch") - pl.col("val_f05"))))
                           .alias("gain_if_blank"))
             .sort("test_share", descending=True))
        print(f"\n=== {country} ({t.height:,} test S1){' - validation = India+US pooled' if country == 'France' else ''} ===")
        print(m.select("cell", pl.col("test_share").round(4), pl.col("val_share").round(4),
                       pl.col("val_nomatch").round(3), pl.col("val_f05").round(3),
                       pl.col("gain_if_blank").round(5)))


if __name__ == "__main__":
    main()
