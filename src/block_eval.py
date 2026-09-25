"""Blocking evaluation on real training data (recall of candidate generation).

For a random sample of training S1 records it builds the key index of the FULL
training S2/S3 pool once per country, then compares blocking configurations:

  coverage   share of true pairs sharing >= 1 key (before ranking)
  recall@k   share of true pairs kept in the final candidate set
  cost       candidates per query

and prints examples of true pairs that no key can find, so the key design can
be improved. Uses only the training data and ground truth.

Usage:  python -m src.block_eval [--n-queries 20000]
"""
import argparse
import time
from pathlib import Path

import polars as pl

from src.aliases import apply_aliases, learn_aliases, load_aliases
from src.blocking import (DEFAULT_CAPS, keys_chunked, pool_key_table, score_rows,
                          select_final)
from src.config import SEED, WORK_DIR, train_paths
from src.io_utils import explode_ground_truth, read_ground_truth
from src.train import load_normalized, log

OLD_CAPS = {0: 200, 1: 300, 2: 300, 3: 300, 4: 150, 5: 100, 6: 0, 7: 0}
BIG_CAPS = {t: 2000 for t in DEFAULT_CAPS}
def _v(caps, weight, rerank, retrieve_k, final_k):
    return dict(caps=caps, weight=weight, rerank=rerank, retrieve_k=retrieve_k, final_k=final_k)


VARIANTS = {
    "v0 old keys, log weights, top10":   _v(OLD_CAPS, "log", "none", 0, 10),
    "v1 new keys, 1/df, top10":          _v(DEFAULT_CAPS, "inv", "none", 0, 10),
    "v2 1/df, top20":                    _v(DEFAULT_CAPS, "inv", "none", 0, 20),
    "v3 1/df, rscore 30->10":            _v(DEFAULT_CAPS, "inv", "rscore", 30, 10),
    "v4 1/df, combo 30->10":             _v(DEFAULT_CAPS, "inv", "combo", 30, 10),
    "v5 1/df, union 5+5 of 30 (default)": _v(DEFAULT_CAPS, "inv", "union", 30, 10),
    "v6 1/df, union 8+8 of 50":          _v(DEFAULT_CAPS, "inv", "union", 50, 16),
    "v7 big caps, union 5+5 of 30":      _v(BIG_CAPS, "inv", "union", 30, 10),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-queries", type=int, default=20_000)
    args = ap.parse_args()

    s1, pool = load_normalized(Path(args.norm_dir), "train")
    gt_pairs = explode_ground_truth(read_ground_truth(train_paths(args.data_dir)["gt"]))
    alias_path = Path(args.model_dir) / "aliases.json"
    aliases = load_aliases(alias_path) if alias_path.exists() else learn_aliases(s1, pool, gt_pairs)
    s1, pool = apply_aliases(s1, aliases), apply_aliases(pool, aliases)
    queries = s1.sample(n=min(args.n_queries, s1.height), seed=SEED + 1)

    results = {v: [0, 0, 0] for v in VARIANTS}   # found, candidates, queries
    cover = {"any key (cap 2000)": 0, "default caps": 0}
    total, missed_frames = 0, []
    for country in sorted(queries["country"].unique().to_list()):
        t0 = time.time()
        q = queries.filter(pl.col("country") == country).with_row_index("idx")
        p = pool.filter(pl.col("country") == country).with_row_index("idx")
        pk = pool_key_table(p, 2000)
        rows = keys_chunked(q).drop("t").join(pk, on="key")
        truth = (gt_pairs.join(q.select("idx", pl.col("entity_id").alias("source1_entity_id")),
                               on="source1_entity_id")
                 .join(p.select(pl.col("idx").alias("pidx"), pl.col("entity_id").alias("matched_id")),
                       on="matched_id").select("idx", "pidx"))
        n_true_all = gt_pairs.filter(pl.col("source1_entity_id").is_in(q["entity_id"].implode())).height
        total += n_true_all
        any_key = truth.join(rows.select("idx", "pidx").unique(), on=["idx", "pidx"], how="semi")
        cover["any key (cap 2000)"] += any_key.height
        capped = score_rows(rows, DEFAULT_CAPS, "inv")
        cover["default caps"] += truth.join(capped, on=["idx", "pidx"], how="semi").height
        src = p.select(pl.col("idx").alias("pidx"), pl.col("entity_id").str.slice(0, 2).alias("source"))
        for name, v in VARIANTS.items():
            sc = select_final(score_rows(rows, v["caps"], v["weight"]).join(src, on="pidx"), q, p, v)
            results[name][0] += truth.join(sc, on=["idx", "pidx"], how="semi").height
            results[name][1] += sc.height
            results[name][2] += q.height
        miss = truth.join(rows.select("idx", "pidx").unique(), on=["idx", "pidx"], how="anti")
        missed_frames.append(
            miss.join(q.select("idx", pl.col("business_name").alias("s1_name"),
                               pl.col("business_address").alias("s1_addr")), on="idx")
            .join(p.select(pl.col("idx").alias("pidx"), pl.col("business_name").alias("cand_name"),
                           pl.col("business_address").alias("cand_addr")), on="pidx")
            .with_columns(pl.lit(country).alias("country")).drop("idx", "pidx"))
        log(f"[{country}] {q.height:,} queries, {truth.height:,} true pairs, "
            f"{rows.height / max(q.height, 1):,.0f} key rows/query, {time.time() - t0:.0f}s")

    print(f"\nTrue pairs in sample: {total:,}")
    for k, v in cover.items():
        print(f"  coverage, {k:22s}: {v / total:.4f}")
    print("\nconfiguration                       recall   cands/query")
    for name, (found, n_c, n_q) in results.items():
        print(f"  {name:34s} {found / total:.4f}   {n_c / max(n_q, 1):5.1f}")
    missed = pl.concat(missed_frames)
    print(f"\n{missed.height:,} true pairs share NO key at all; examples:")
    with pl.Config(tbl_rows=25, fmt_str_lengths=55, tbl_width_chars=240):
        print(missed.sample(n=min(25, missed.height), seed=SEED))


if __name__ == "__main__":
    main()
