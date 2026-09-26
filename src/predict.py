"""Predict matches for the test set and write the two submission files.

Stages:
  1. load normalized test data, apply the aliases learned on train
  2. blocking for EVERY test S1 record vs the test S2/S3 pool (cached)
  3. features + LightGBM probabilities, in chunks of S1 records (cached)
  4. exclusivity + decision rule tuned on validation
  5. write output/matching_results.tsv and output/candidate_pairs.tsv
     (candidate_pairs = exactly the pairs the model scored), check them
     against every submission rule, and optionally run the official validator.

Usage:  python -m src.predict [--validator /path/to/validate_submission.py]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.aliases import apply_aliases, load_aliases
from src.blocking import generate_candidates
from src.config import DATA_DIR, WORK_DIR
from src.decision import G, decide, to_lists
from src.features import FEATURES, compute_features_chunked, token_idf
from src.io_utils import write_id_lists
from src.train import load_normalized, log


def check_outputs(match: pl.DataFrame, cand: pl.DataFrame, s1_ids: pl.Series,
                  pool_ids: pl.Series) -> list:
    """Our own copy of the submission rules; returns a list of problems."""
    problems = []
    for name, df in (("matching", match), ("candidates", cand)):
        if df.height != s1_ids.len() or df[G].n_unique() != df.height:
            problems.append(f"{name}: rows {df.height:,} vs S1 {s1_ids.len():,} (or duplicates)")
        if not df[G].is_in(s1_ids.implode()).all():
            problems.append(f"{name}: unknown S1 ids")
        ids = df.select(pl.col("ids").list.explode()).drop_nulls()["ids"]
        if not ids.is_in(pool_ids.implode()).all():
            problems.append(f"{name}: ids not in test S2/S3")
        if (df["ids"].list.len() != df["ids"].list.unique().list.len()).any():
            problems.append(f"{name}: duplicate ids inside a list")
    sub = match.join(cand, on=G, suffix="_c").filter(
        pl.col("ids").list.set_difference(pl.col("ids_c")).list.len() > 0)
    if sub.height:
        problems.append(f"{sub.height} S1 rows have matches outside their candidates")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model"))
    ap.add_argument("--out-dir", default=str(Path(WORK_DIR) / "output"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--validator", default=None, help="path to utils/validate_submission.py")
    ap.add_argument("--force", action="store_true", help="recompute cached stages")
    ap.add_argument("--rescore", action="store_true",
                    help="reuse cached test candidates but recompute features + scores "
                         "(use after retraining the model)")
    ap.add_argument("--tau", type=float, default=None,
                    help="override the tuned rule with a plain probability threshold")
    ap.add_argument("--post", default="none",
                    help="house-number conflict rule from src.postprocess.VARIANTS (e.g. group)")
    ap.add_argument("--blank-country", nargs="*", default=[],
                    help="PROBE ONLY: predict no matches for these countries (measures their score)")
    args = ap.parse_args()

    norm_dir, mdir, out = Path(args.norm_dir), Path(args.model_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    params = json.loads((mdir / "params.json").read_text())
    model = lgb.Booster(model_file=str(mdir / "model.txt"))
    aliases = load_aliases(mdir / "aliases.json")

    log("loading normalized test data")
    s1, pool = load_normalized(norm_dir, "test")
    s1, pool = apply_aliases(s1, aliases), apply_aliases(pool, aliases)
    log(f"test: {s1.height:,} S1, {pool.height:,} S2/S3; countries "
        f"{s1.group_by('country').len().sort('len', descending=True).rows()}")

    cand_path = mdir / "test_cands.parquet"
    if args.force or not cand_path.exists():
        log("blocking (all test S1 vs test S2/S3 pool)")
        generate_candidates(s1, pool, top_k=params["top_k"], log=log).write_parquet(cand_path)
    cands = pl.read_parquet(cand_path)
    log(f"candidates: {cands.height:,} pairs ({cands.height / s1.height:.1f} per S1)")

    scored_path = mdir / "test_scored.parquet"
    if args.force or args.rescore or not scored_path.exists():
        log("features + scoring")
        idf = token_idf(s1, pool)
        parts = []
        for f in compute_features_chunked(cands, s1, pool, idf, log=log):
            parts.append(f.select(G, "cand_id", "country").with_columns(
                pl.Series("p", model.predict(f.select(FEATURES).to_numpy()))))
        pl.concat(parts).write_parquet(scored_path)
    scored = pl.read_parquet(scored_path)

    rule = params["decision"] if args.tau is None else {"rule": "threshold", "tau": args.tau}
    log(f"decision rule: {rule}")
    if args.post != "none":
        from src.postprocess import apply_post, numbers_frame
        log(f"post-processing: {args.post}")
        scored = apply_post(scored, numbers_frame(s1, G, "nums1"),
                            numbers_frame(pool, "cand_id", "nums2"), args.post)
    if args.blank_country:
        log(f"PROBE: no matches predicted for {args.blank_country}")
        scored = scored.filter(~pl.col("country").is_in(args.blank_country))
    s1_ids = s1["entity_id"]
    match = to_lists(decide(scored, rule), s1_ids.to_list())
    cand_lists = to_lists(cands.select(G, "cand_id"), s1_ids.to_list())

    problems = check_outputs(match, cand_lists, s1_ids, pool["entity_id"])
    if problems:
        for p in problems:
            log("PROBLEM: " + p)
        sys.exit(1)
    write_id_lists(match, out / "matching_results.tsv", "matched_entity_ids")
    write_id_lists(cand_lists, out / "candidate_pairs.tsv", "candidate_entity_ids")
    n = match["ids"].list.len()
    stats = (match.join(s1.select(pl.col("entity_id").alias(G), "country"), on=G)
             .group_by("country").agg(pl.len().alias("s1"),
                                      pl.col("ids").list.len().mean().round(2).alias("pred_per_s1"),
                                      (pl.col("ids").list.len() == 0).mean().round(3).alias("empty_share")))
    log(f"predicted {n.sum():,} matches; {(n == 0).mean():.3%} S1 with no match")
    print(stats)

    if args.validator:
        log("running the official validator")
        r = subprocess.run([sys.executable, args.validator,
                            "--matching", str(out / "matching_results.tsv"),
                            "--candidate", str(out / "candidate_pairs.tsv"),
                            "--test-dir", str(Path(args.data_dir) / "test")],
                           capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-1000:])
    log(f"done in {(time.time() - t_all) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
