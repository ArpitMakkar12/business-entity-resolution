"""Train the matching model on a sample of training S1 entities.

Stages (each cached under WORK_DIR/model/, re-run with --force):
  1. learn aliases from ground-truth pairs          -> aliases.json
  2. blocking for sampled S1 records vs FULL pool   -> train_cands.parquet
     (the full training S2/S3 pool keeps negatives as hard as on test)
  3. pair features + labels                         -> train_feats.parquet
  4. LightGBM on the train split, early stopping on the valid split
                                                    -> model.txt
  5. decision rule tuned for macro F0.5 on valid    -> params.json
     plus a validation report (blocking recall ceiling, F0.5 per country,
     singletons vs multi-match entities)             -> report.json

Train/valid split is by S1 entity, so no entity appears in both.

Usage:  python -m src.train [--n-train 300000] [--n-valid 100000] [--force]
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.aliases import apply_aliases, learn_aliases, load_aliases, save_aliases
from src.blocking import blocking_recall, generate_candidates
from src.config import N_JOBS, SEED, WORK_DIR, train_paths
from src.decision import G, decide, f05_per_entity, to_lists, tune_decision
from src.features import FEATURES, compute_features_chunked, token_idf
from src.io_utils import explode_ground_truth, read_ground_truth

LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 127,
    "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": SEED,
    "num_threads": N_JOBS,
}


def load_normalized(norm_dir: Path, split: str):
    """Return (s1, pool) normalized frames for 'train' or 'test'."""
    s1 = pl.read_parquet(norm_dir / f"{split}_s1.parquet")
    pool = pl.concat([pl.read_parquet(norm_dir / f"{split}_s2.parquet"),
                      pl.read_parquet(norm_dir / f"{split}_s3.parquet")])
    return s1, pool


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model"))
    ap.add_argument("--data-dir", default=None, help="challenge dataset dir (for ground truth)")
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--n-valid", type=int, default=100_000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--force", action="store_true", help="recompute cached stages")
    args = ap.parse_args()

    norm_dir, mdir = Path(args.norm_dir), Path(args.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    log("loading normalized training data")
    s1, pool = load_normalized(norm_dir, "train")
    gt = read_ground_truth(train_paths(args.data_dir)["gt"])
    gt_pairs = explode_ground_truth(gt)

    # 1. aliases -------------------------------------------------------------
    alias_path = mdir / "aliases.json"
    if args.force or not alias_path.exists():
        aliases = learn_aliases(s1, pool, gt_pairs)
        save_aliases(aliases, alias_path)
    aliases = load_aliases(alias_path)
    log(f"aliases: {len(aliases):,} learned, e.g. {dict(list(aliases.items())[:6])}")
    s1, pool = apply_aliases(s1, aliases), apply_aliases(pool, aliases)

    # sample S1 entities (split by entity) -----------------------------------
    ids = s1.select("entity_id").sample(fraction=1.0, shuffle=True, seed=SEED)["entity_id"]
    n_tr = min(args.n_train, int(len(ids) * 0.8))
    n_va = min(args.n_valid, len(ids) - n_tr)
    train_ids, valid_ids = ids[:n_tr], ids[n_tr:n_tr + n_va]
    query = s1.filter(pl.col("entity_id").is_in(pl.concat([train_ids, valid_ids]).implode()))
    log(f"sampled {n_tr:,} train + {n_va:,} valid S1 entities")

    # 2. blocking ------------------------------------------------------------
    cand_path = mdir / "train_cands.parquet"
    if args.force or not cand_path.exists():
        log("blocking (sampled S1 vs full S2/S3 pool)")
        cands = generate_candidates(query, pool, top_k=args.top_k, log=log)
        cands.write_parquet(cand_path)
    cands = pl.read_parquet(cand_path)
    rec = blocking_recall(cands, gt_pairs, query["entity_id"])
    log(f"blocking recall on sample: {rec}")

    # 3. features ------------------------------------------------------------
    feat_path = mdir / "train_feats.parquet"
    if args.force or not feat_path.exists():
        log("computing features")
        idf = token_idf(s1, pool)
        parts = list(compute_features_chunked(cands, query, pool, idf, log=log))
        feats = pl.concat(parts).join(
            gt_pairs.rename({"matched_id": "cand_id"}).with_columns(pl.lit(1, pl.Int8).alias("y")),
            on=["source1_entity_id", "cand_id"], how="left").with_columns(pl.col("y").fill_null(0))
        feats.write_parquet(feat_path)
    feats = pl.read_parquet(feat_path)
    is_tr = pl.col("source1_entity_id").is_in(train_ids.implode())
    tr, va = feats.filter(is_tr), feats.filter(~is_tr)
    log(f"pairs: train {tr.height:,} (pos {tr['y'].mean():.3f}), valid {va.height:,}")

    # 4. LightGBM ------------------------------------------------------------
    log("training LightGBM")
    dtr = lgb.Dataset(tr.select(FEATURES).to_numpy(), tr["y"].to_numpy(), feature_name=FEATURES)
    dva = lgb.Dataset(va.select(FEATURES).to_numpy(), va["y"].to_numpy(), reference=dtr)
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    model.save_model(str(mdir / "model.txt"))
    imp = sorted(zip(FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])
    log("top features: " + ", ".join(f"{n}={v:.0f}" for n, v in imp[:12]))

    # 5. decision rule + validation report -----------------------------------
    scored = va.select(G, "cand_id", "country", "y").with_columns(
        pl.Series("p", model.predict(va.select(FEATURES).to_numpy())))
    truth = gt.filter(pl.col(G).is_in(valid_ids.implode()))
    log("tuning decision rule on validation (macro F0.5)")
    params = tune_decision(scored, truth, log=log)
    per = f05_per_entity(to_lists(decide(scored, params), truth[G].to_list()), truth)
    per = per.join(s1.select(pl.col("entity_id").alias(G), "country"), on=G)
    by_country = {c: round(v, 5) for c, v in per.group_by("country").agg(pl.col("f05").mean()).iter_rows()}
    by_type = {("singleton" if k else "has_matches"): round(v, 5)
               for k, v in per.group_by("is_singleton").agg(pl.col("f05").mean()).iter_rows()}
    report = {"blocking": rec, "valid_f05": params["valid_f05"], "by_country": by_country,
              "by_type": by_type, "mean_precision": round(per["precision"].mean(), 4),
              "mean_recall": round(per["recall"].mean(), 4), "best_iteration": model.best_iteration,
              "decision": params, "n_train_s1": n_tr, "n_valid_s1": n_va, "top_k": args.top_k}
    (mdir / "params.json").write_text(json.dumps({"decision": params, "top_k": args.top_k,
                                                   "features": FEATURES}, indent=2))
    (mdir / "report.json").write_text(json.dumps(report, indent=2))
    log(f"VALIDATION macro F0.5 = {params['valid_f05']:.5f}  by country {by_country}  {by_type}")
    log(f"done in {(time.time() - t_all) / 60:.1f} min -> {mdir}")


if __name__ == "__main__":
    main()
