"""Honorific-prefix fix ("Shri", "Sri", "Smt", "Mr", "Dr") without re-running the pipeline.

The test audit showed real copies scored 0.3-0.7 and missed only because of an
added honorific ("Shri Gem Life Private Ltd" vs S1 "Gem Life Private Limited",
same address). These prefixes are 1.2-1.5x more frequent in the test pool than in
the training pool. This script strips them (from S1 and S2/S3 alike, only when
another word follows), recomputes the normalized views of the affected records,
and recomputes features + probabilities for every S1 entity whose candidate list
contains an affected record (whole candidate lists, so group features are exact).
Blocking and candidate lists are not changed.

  validate  measures the effect on the validation split of a trained model
            (same split as src.train) - run this first
  test      writes patched copies of the model directories (suffix 'h'):
            model.txt, params.json, aliases.json, test_cands.parquet and a
            test_scored.parquet where the affected S1 entities are rescored;
            src.predict then runs on them unchanged

Usage:
  python -m src.patch_names validate --model-dir WORK_DIR/model_v3
  python -m src.patch_names test --model-dirs WORK_DIR/model_v3 WORK_DIR/model_v4
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import polars as pl

import src.normalize as N
from src.aliases import apply_aliases, load_aliases
from src.config import SEED, WORK_DIR, train_paths
from src.decision import G, decide, f05_per_entity, to_lists
from src.features import FEATURES, add_name_freq, compute_features_chunked, token_idf
from src.io_utils import explode_ground_truth, read_ground_truth
from src.normalize import VIEW_COLUMNS, add_views
from src.postprocess import apply_post, numbers_frame
from src.train import load_normalized, log

HONORIFICS = ["shri", "sri", "smt", "mr", "mrs", "ms", "dr"]


def patch(df: pl.DataFrame, aliases: dict, words, countries) -> tuple:
    """Re-normalize records whose name starts with an honorific -> (frame, patched ids)."""
    m = (pl.col("name_tokens").list.first().is_in(words) & (pl.col("name_tokens").list.len() >= 2)
         & pl.col("country").is_in(countries))
    sub = df.filter(m)
    if sub.height == 0:
        return df, pl.Series("entity_id", [], dtype=pl.Utf8)
    raw_cols = [c for c in sub.columns if c not in VIEW_COLUMNS and c != "name_freq"]
    N.STRIP_PREFIX = list(words)
    try:
        new = add_views(sub.select(raw_cols))
    finally:
        N.STRIP_PREFIX = []
    new = apply_aliases(new, aliases).select(df.columns)
    return pl.concat([df.filter(~m), new]), sub["entity_id"]


def affected_s1(cands: pl.DataFrame, p_s1: pl.Series, p_pool: pl.Series) -> pl.Series:
    a = cands.filter(pl.col("cand_id").is_in(p_pool.implode())).select(G).unique()[G]
    return pl.concat([a, p_s1.rename(G)]).unique()


def validate(args):
    mdir, ndir = Path(args.model_dir), Path(args.norm_dir)
    t0 = time.time()
    tau = json.loads((mdir / "params.json").read_text())["decision"].get("tau", 0.7)
    aliases = load_aliases(mdir / "aliases.json")
    s1, pool = load_normalized(ndir, "train")
    s1, pool = apply_aliases(s1, aliases), apply_aliases(pool, aliases)
    if (mdir / "siblings.parquet").exists():
        pool = pl.concat([pool, pl.read_parquet(mdir / "siblings.parquet").select(pool.columns)])
    s1p, ps1 = patch(s1, aliases, args.words, args.countries)
    poolp, ppool = patch(pool, aliases, args.words, args.countries)
    log(f"patched records: S1 {len(ps1):,}, S2/S3 {len(ppool):,}")
    s1p, poolp = add_name_freq(s1p), add_name_freq(poolp)
    ids = (pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id"])
           .sample(fraction=1.0, shuffle=True, seed=SEED)["entity_id"])
    n_tr = min(args.n_train, int(len(ids) * 0.8))
    valid_ids = ids[n_tr:n_tr + min(args.n_valid, len(ids) - n_tr)]
    cands = pl.read_parquet(mdir / "train_cands.parquet").filter(pl.col(G).is_in(valid_ids.implode()))
    aff = affected_s1(cands, ps1, ppool)
    aff = aff.filter(aff.is_in(valid_ids.implode()))
    log(f"validation S1 affected: {len(aff):,} of {len(valid_ids):,}")
    model = lgb.Booster(model_file=str(mdir / "model.txt"))
    old = pl.read_parquet(mdir / "train_feats.parquet").filter(pl.col(G).is_in(valid_ids.implode()))
    old = old.select(G, "cand_id", "country", "y").with_columns(
        pl.Series("p", model.predict(old.select(FEATURES).to_numpy())))
    idf = token_idf(s1p, poolp)
    q = s1p.filter(pl.col("entity_id").is_in(aff.implode()))
    ca = cands.filter(pl.col(G).is_in(aff.implode()))
    gt = explode_ground_truth(read_ground_truth(train_paths(args.data_dir)["gt"]))
    parts = []
    for f in compute_features_chunked(ca, q, poolp, idf, log=log):
        parts.append(f.select(G, "cand_id", "country").with_columns(
            pl.Series("p", model.predict(f.select(FEATURES).to_numpy()))))
    new = (pl.concat(parts).join(gt.rename({"matched_id": "cand_id"}).with_columns(pl.lit(1, pl.Int8).alias("y")),
                                 on=[G, "cand_id"], how="left").with_columns(pl.col("y").fill_null(0)))
    patched = pl.concat([old.filter(~pl.col(G).is_in(aff.implode())),
                         new.select(old.columns)])
    truth = read_ground_truth(train_paths(args.data_dir)["gt"]).filter(pl.col(G).is_in(valid_ids.implode()))
    s1n = numbers_frame(s1, G, "nums1")
    pooln = numbers_frame(pool, "cand_id", "nums2")
    cmp = old.join(new.select(G, "cand_id", pl.col("p").alias("p_new")), on=[G, "cand_id"])
    up = cmp.filter((pl.col("p") < tau) & (pl.col("p_new") >= tau))
    down = cmp.filter((pl.col("p") >= tau) & (pl.col("p_new") < tau))
    print(f"\npairs crossing tau {tau}: up {up.height:,} (true {int(up['y'].sum()):,}), "
          f"down {down.height:,} (true {int(down['y'].sum()):,})")
    for label, sc in (("before", old), ("after (honorifics stripped)", patched)):
        sc2 = apply_post(sc.select(G, "cand_id", "country", "p"), s1n, pooln, "conflict_p90")
        per = f05_per_entity(to_lists(decide(sc2, {"rule": "threshold", "tau": tau}), truth[G].to_list()), truth)
        per = per.join(s1.select(pl.col("entity_id").alias(G), "country"), on=G)
        by = dict(per.group_by("country").agg(pl.col("f05").mean()).iter_rows())
        print(f"validation macro F0.5 {label:28s}: {per['f05'].mean():.5f}  "
              + "  ".join(f"{k} {v:.5f}" for k, v in sorted(by.items())))
    log(f"done in {(time.time() - t0) / 60:.1f} min")


def test(args):
    mdirs = [Path(d) for d in args.model_dirs]
    ndir = Path(args.norm_dir)
    t0 = time.time()
    aliases = load_aliases(mdirs[0] / "aliases.json")
    for d in mdirs[1:]:
        if load_aliases(d / "aliases.json") != aliases:
            log(f"WARNING: {d} has different aliases; using those of {mdirs[0]}")
    s1, pool = load_normalized(ndir, "test")
    s1, pool = apply_aliases(s1, aliases), apply_aliases(pool, aliases)
    s1p, ps1 = patch(s1, aliases, args.words, args.countries)
    poolp, ppool = patch(pool, aliases, args.words, args.countries)
    log(f"patched records: S1 {len(ps1):,}, S2/S3 {len(ppool):,}")
    s1p, poolp = add_name_freq(s1p), add_name_freq(poolp)
    cands = pl.read_parquet(mdirs[0] / "test_cands.parquet")
    aff = affected_s1(cands, ps1, ppool)
    log(f"test S1 affected: {len(aff):,}")
    idf = token_idf(s1p, poolp)
    models = [lgb.Booster(model_file=str(d / "model.txt")) for d in mdirs]
    q = s1p.filter(pl.col("entity_id").is_in(aff.implode()))
    ca = cands.filter(pl.col(G).is_in(aff.implode()))
    parts = []
    for f in compute_features_chunked(ca, q, poolp, idf, log=log):
        X = f.select(FEATURES).to_numpy()
        parts.append(f.select(G, "cand_id", "country").with_columns(
            [pl.Series(f"p{i}", m.predict(X)) for i, m in enumerate(models)]))
    new = pl.concat(parts)
    names1 = s1.select(pl.col("entity_id").alias(G), pl.col("business_name").alias("s1_name"),
                       pl.col("business_address").alias("s1_addr"))
    names2 = pool.select(pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("rec_name"),
                         pl.col("business_address").alias("rec_addr"))
    for i, d in enumerate(mdirs):
        dst = Path(str(d) + args.suffix)
        dst.mkdir(parents=True, exist_ok=True)
        for fn in ("model.txt", "params.json", "aliases.json", "test_cands.parquet"):
            if not (dst / fn).exists():
                shutil.copy(d / fn, dst / fn)
        old = pl.read_parquet(d / "test_scored.parquet")
        tau = json.loads((d / "params.json").read_text())["decision"].get("tau", 0.7)
        cmp = old.join(new.select(G, "cand_id", pl.col(f"p{i}").alias("p_new")), on=[G, "cand_id"])
        up = cmp.filter((pl.col("p") < tau) & (pl.col("p_new") >= tau))
        down = cmp.filter((pl.col("p") >= tau) & (pl.col("p_new") < tau))
        log(f"{d.name}: pairs crossing tau {tau}: up {up.height:,}, down {down.height:,}")
        out = pl.concat([old.filter(~pl.col(G).is_in(aff.implode())),
                         new.select(G, "cand_id", "country", pl.col(f"p{i}").alias("p")).select(old.columns)])
        assert out.height == old.height, (out.height, old.height)
        out.write_parquet(dst / "test_scored.parquet")
        log(f"  -> {dst}")
        if i == 0:
            pl.Config.set_tbl_rows(30)
            pl.Config.set_tbl_cols(-1)
            pl.Config.set_tbl_width_chars(250)
            pl.Config.set_fmt_str_lengths(34)
            print("\nexamples of pairs newly above tau (random 25):")
            print(up.sample(n=min(25, up.height), seed=1).join(names1, on=G).join(names2, on="cand_id")
                  .select("s1_name", "s1_addr", "rec_name", "rec_addr", pl.col("p").round(3),
                          pl.col("p_new").round(3)))
            if down.height:
                print("\nexamples of pairs newly below tau (random 15):")
                print(down.sample(n=min(15, down.height), seed=2).join(names1, on=G).join(names2, on="cand_id")
                      .select("s1_name", "s1_addr", "rec_name", "rec_addr", pl.col("p").round(3),
                              pl.col("p_new").round(3)))
    log(f"done in {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["validate", "test"])
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model_v3"))
    ap.add_argument("--model-dirs", nargs="*", default=[str(Path(WORK_DIR) / "model_v3"),
                                                        str(Path(WORK_DIR) / "model_v4")])
    ap.add_argument("--suffix", default="h")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--n-valid", type=int, default=100_000)
    ap.add_argument("--words", nargs="*", default=HONORIFICS)
    ap.add_argument("--countries", nargs="*", default=["India"])
    args = ap.parse_args()
    validate(args) if args.cmd == "validate" else test(args)


if __name__ == "__main__":
    main()
