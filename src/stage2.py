"""Stage 2: group-consistency re-scoring of the stage-1 pair probabilities.

Stage 1 scores every (S1, candidate) pair on its own. But the true matches of
one S1 entity are noisy copies of the *same* business and agree with each
other, while a sibling distractor (neighbouring house number, one name word
changed) or an ambiguous address-less record disagrees with that group. Stage 2
re-scores each candidate with features that compare it to the *confident*
matches ("anchors", stage-1 p >= 0.9) of the same S1 and to competing S1s:

  group     stage-1 p, rank, number of other anchors, best / summed p of others
  numbers   does the candidate carry the S1 house number / the anchors' numbers,
            how many anchors share its number, size and mean p of the cluster of
            candidates sharing its number or its exact name
  anchors   best name and address token-set similarity to any other anchor
  cross-S1  best stage-1 p of the same candidate for a *different* S1
  pair      house-number conflict, candidate address empty

Training uses the validation split of src.train only (stage-1 predictions there
are out-of-sample): 5-fold cross-validation grouped by S1 entity gives an
honest estimate, then a final model is fit on all validation entities.

Usage:
  python -m src.stage2 fit                       # CV report + model -> WORK_DIR/model/stage2*
  python -m src.stage2 predict --out-dir DIR [--validator tools/validate_submission.py]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from src.aliases import apply_aliases, load_aliases
from src.config import N_JOBS, SEED, WORK_DIR, train_paths
from src.decision import G, decide, f05_macro, to_lists
from src.features import FEATURES
from src.io_utils import read_ground_truth, write_id_lists
from src.postprocess import apply_post, numbers_frame
from src.train import log

P_FLOOR = 0.01       # pairs below this stage-1 probability keep p2 = 0
ANCHOR_P = 0.9
S2_FEATURES = [
    "p1", "p_rank", "grp_n", "n_anchor_other", "p_max_other", "p_sum_other", "p_gap_top",
    "s1_num_in_cand", "cand_has_num", "s1_has_num", "num_conflict", "cand_addr_empty",
    "n_anchor_same_num", "frac_anchor_same_num", "anchor_has_num",
    "cl_num_n", "cl_num_pmean", "cl_name_n", "cl_name_pmean",
    "anc_name_max", "anc_addr_max", "anc_name_mean",
    "p_best_other_s1", "is_best_s1", "n_s1_claims",
]
LGB2 = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "verbose": -1, "seed": SEED, "num_threads": N_JOBS}
REC_COLS = ["entity_id", "name_core", "name_concat", "addr_clean", "addr_numbers", "addr_empty"]


def records(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(REC_COLS)


def group_features(all_scored: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """Stage-2 features for pairs with p >= P_FLOOR.

    all_scored: (source1_entity_id, cand_id, p) for *all* candidates (used for the
    cross-S1 features); s1 / pool: normalized records (REC_COLS).
    """
    cross = (all_scored.filter(pl.col("p") >= P_FLOOR)
             .with_columns(pl.col("p").rank("ordinal", descending=True).over("cand_id").alias("_r"),
                           pl.len().over("cand_id").alias("n_s1_claims")))
    d = cross.with_columns(pl.col("p").alias("p1"),
                           (pl.col("_r") == 1).cast(pl.Int8).alias("is_best_s1"))
    # best p of the same candidate for a different S1
    top2 = (cross.group_by("cand_id").agg(pl.col("p").top_k(2).alias("_t")))
    d = d.join(top2, on="cand_id", how="left").with_columns(
        pl.when(pl.col("_r") == 1).then(pl.col("_t").list.get(1, null_on_oob=True))
        .otherwise(pl.col("_t").list.get(0)).fill_null(0.0).alias("p_best_other_s1")).drop("_t", "_r", "p")

    a = s1.select(pl.col("entity_id").alias(G), pl.col("addr_numbers").alias("n1"),
                  pl.col("name_core").alias("s1_name"))
    b = pool.select(pl.col("entity_id").alias("cand_id"), pl.col("addr_numbers").alias("n2"),
                    pl.col("name_core").alias("c_name"), pl.col("name_concat").alias("c_concat"),
                    pl.col("addr_clean").alias("c_addr"), pl.col("addr_empty").alias("c_empty"))
    d = d.join(a, on=G, how="left").join(b, on="cand_id", how="left")
    L = lambda c: pl.col(c).list.len().fill_null(0)  # noqa: E731
    d = d.with_columns(
        pl.col("n2").list.first().fill_null("").alias("_n2first"),
        pl.col("n2").list.contains(pl.col("n1").list.first()).fill_null(False).cast(pl.Int8)
        .alias("s1_num_in_cand"),
        (L("n2") > 0).cast(pl.Int8).alias("cand_has_num"),
        (L("n1") > 0).cast(pl.Int8).alias("s1_has_num"),
        ((L("n1") > 0) & (L("n2") > 0)
         & (pl.col("n1").list.set_intersection(pl.col("n2")).list.len().fill_null(0) == 0))
        .cast(pl.Int8).alias("num_conflict"),
        pl.col("c_empty").fill_null(True).cast(pl.Int8).alias("cand_addr_empty"),
        (pl.col("p1") >= ANCHOR_P).cast(pl.Int32).alias("_anc"),
    )
    d = d.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over(G).cast(pl.Float32).alias("p_rank"),
        pl.len().over(G).cast(pl.Float32).alias("grp_n"),
        (pl.col("_anc").sum().over(G) - pl.col("_anc")).cast(pl.Float32).alias("n_anchor_other"),
        (pl.col("p1").sum().over(G) - pl.col("p1")).alias("p_sum_other"),
        (pl.col("p1").max().over(G) - pl.col("p1")).alias("p_gap_top"),
        pl.col("p1").top_k(2).over(G, mapping_strategy="join").alias("_top2"),
        pl.when(pl.col("_n2first") != "").then(pl.len().over([G, "_n2first"]) - 1).otherwise(0)
        .cast(pl.Float32).alias("cl_num_n"),
        pl.when(pl.col("_n2first") != "")
        .then((pl.col("p1").sum().over([G, "_n2first"]) - pl.col("p1"))
              / (pl.len().over([G, "_n2first"]) - 1).clip(lower_bound=1)).otherwise(0.0)
        .alias("cl_num_pmean"),
        (pl.len().over([G, "c_concat"]) - 1).cast(pl.Float32).alias("cl_name_n"),
        ((pl.col("p1").sum().over([G, "c_concat"]) - pl.col("p1"))
         / (pl.len().over([G, "c_concat"]) - 1).clip(lower_bound=1)).alias("cl_name_pmean"),
    ).with_columns(
        pl.when(pl.col("p1") >= pl.col("_top2").list.first())
        .then(pl.col("_top2").list.get(1, null_on_oob=True)).otherwise(pl.col("_top2").list.first())
        .fill_null(0.0).alias("p_max_other"),
    ).drop("_top2").with_row_index("rid")

    # anchor-relative features: candidate vs every *other* anchor of the same S1
    anc = d.filter(pl.col("_anc") == 1).select(
        G, pl.col("rid").alias("arid"), pl.col("n2").alias("an2"), pl.col("c_name").alias("aname"),
        pl.col("c_addr").alias("aaddr"))
    x = d.select(G, "rid", "n2", "c_name", "c_addr").join(anc, on=G).filter(pl.col("rid") != pl.col("arid"))
    if x.height:
        ns = cpdist(x["c_name"].fill_null("").to_list(), x["aname"].fill_null("").to_list(),
                    scorer=fuzz.token_set_ratio, workers=N_JOBS, dtype=np.float32) / 100
        ad = cpdist(x["c_addr"].fill_null("").to_list(), x["aaddr"].fill_null("").to_list(),
                    scorer=fuzz.token_set_ratio, workers=N_JOBS, dtype=np.float32) / 100
        x = x.with_columns(pl.Series("ns", ns), pl.Series("ad", ad),
                           (pl.col("n2").list.set_intersection(pl.col("an2")).list.len().fill_null(0) > 0)
                           .alias("same_num"),
                           (pl.col("an2").list.len().fill_null(0) > 0).alias("anc_num"))
        agg = x.group_by("rid").agg(
            pl.col("ns").max().alias("anc_name_max"), pl.col("ns").mean().alias("anc_name_mean"),
            pl.col("ad").max().alias("anc_addr_max"),
            pl.col("same_num").sum().cast(pl.Float32).alias("n_anchor_same_num"),
            pl.col("anc_num").any().cast(pl.Int8).alias("anchor_has_num"))
        d = d.join(agg, on="rid", how="left")
    else:
        d = d.with_columns([pl.lit(None, pl.Float32).alias(c) for c in
                            ("anc_name_max", "anc_name_mean", "anc_addr_max", "n_anchor_same_num")]
                           + [pl.lit(None, pl.Int8).alias("anchor_has_num")])
    d = d.with_columns(
        pl.col("n_anchor_same_num").fill_null(0.0),
        pl.col("anchor_has_num").fill_null(0),
        (pl.col("n_anchor_same_num").fill_null(0.0) / pl.col("n_anchor_other").clip(lower_bound=1))
        .alias("frac_anchor_same_num"),
        pl.col("anc_name_max").fill_null(-1.0), pl.col("anc_name_mean").fill_null(-1.0),
        pl.col("anc_addr_max").fill_null(-1.0),
        pl.col("n_s1_claims").cast(pl.Float32),
    )
    return d.select([G, "cand_id"] + [pl.col(f).cast(pl.Float32) for f in S2_FEATURES])


def _load_split(ndir: Path, split: str, mdir: Path):
    s1 = pl.read_parquet(ndir / f"{split}_s1.parquet")
    pool = pl.concat([pl.read_parquet(ndir / f"{split}_s{k}.parquet") for k in (2, 3)])
    if split == "train" and (mdir / "siblings.parquet").exists():
        sib = pl.read_parquet(mdir / "siblings.parquet")
        pool = pl.concat([pool, sib.select(pool.columns)])
    al = load_aliases(mdir / "aliases.json")
    return apply_aliases(s1, al), apply_aliases(pool, al)


def _grid(scored: pl.DataFrame, truth: pl.DataFrame, s1n, pooln, taus, posts):
    ids = truth[G].to_list()
    res = []
    for post in posts:
        sc = apply_post(scored, s1n, pooln, post) if post != "none" else scored
        for tau in taus:
            f = f05_macro(to_lists(decide(sc, {"rule": "threshold", "tau": tau}), ids), truth)
            res.append((f, tau, post))
    return sorted(res, reverse=True)


def fit(args):
    mdir, ndir = Path(args.model_dir), Path(args.norm_dir)
    t0 = time.time()
    s1_ids = pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id"])
    ids = s1_ids.select("entity_id").sample(fraction=1.0, shuffle=True, seed=SEED)["entity_id"]
    n_tr = min(args.n_train, int(len(ids) * 0.8))
    valid_ids = ids[n_tr:n_tr + min(args.n_valid, len(ids) - n_tr)]
    feats = pl.read_parquet(mdir / "train_feats.parquet").filter(pl.col(G).is_in(valid_ids.implode()))
    model = lgb.Booster(model_file=str(mdir / "model.txt"))
    scored = feats.select(G, "cand_id", "country", "y").with_columns(
        pl.Series("p", model.predict(feats.select(FEATURES).to_numpy())))
    truth = read_ground_truth(train_paths(args.data_dir)["gt"]).filter(pl.col(G).is_in(valid_ids.implode()))
    log(f"validation: {truth.height:,} S1, {scored.height:,} pairs; building stage-2 features")
    s1, pool = _load_split(ndir, "train", mdir)
    X = group_features(scored.select(G, "cand_id", "p"), records(s1), records(pool))
    X = X.join(scored.select(G, "cand_id", "y", "country"), on=[G, "cand_id"])
    log(f"stage-2 pairs: {X.height:,} (p1 >= {P_FLOOR}), positives {int(X['y'].sum()):,}")

    fold = (X[G].hash(seed=3) % 5).to_numpy()
    oof = np.zeros(X.height)
    iters = []
    Xn, y = X.select(S2_FEATURES).to_numpy(), X["y"].to_numpy()
    for k in range(5):
        tr, va = fold != k, fold == k
        m = lgb.train(LGB2, lgb.Dataset(Xn[tr], y[tr], feature_name=S2_FEATURES), 2000,
                      valid_sets=[lgb.Dataset(Xn[va], y[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(Xn[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"  fold {k}: best iteration {m.best_iteration}")
    s1n = numbers_frame(s1, G, "nums1")
    pooln = numbers_frame(pool, "cand_id", "nums2")
    st1 = scored.select(G, "cand_id", "country", "p")
    st2 = X.select(G, "cand_id", "country").with_columns(pl.Series("p", oof))
    taus = [0.4, 0.5, 0.6, 0.7, 0.8]
    r1 = _grid(st1, truth, s1n, pooln, [0.7], ["none", "conflict_p90"])
    r2 = _grid(st2, truth, s1n, pooln, taus, ["none", "conflict_p90"])
    log("stage 1 (reference): " + ", ".join(f"{p}@{t}: {f:.5f}" for f, t, p in r1))
    log("stage 2 (5-fold CV) top: " + ", ".join(f"{p}@{t}: {f:.5f}" for f, t, p in r2[:6]))
    best_f, best_tau, best_post = r2[0]
    final = lgb.train(LGB2, lgb.Dataset(Xn, y, feature_name=S2_FEATURES), int(np.mean(iters) * 1.1))
    final.save_model(str(mdir / "stage2_model.txt"))
    imp = sorted(zip(S2_FEATURES, final.feature_importance("gain")), key=lambda t: -t[1])
    log("stage-2 top features: " + ", ".join(f"{n}={v:.0f}" for n, v in imp[:10]))
    rep = {"stage1_valid_f05": {f"{p}@{t}": round(f, 5) for f, t, p in r1},
           "stage2_cv_f05": round(best_f, 5), "tau": best_tau, "post": best_post,
           "grid": [(round(f, 5), t, p) for f, t, p in r2]}
    (mdir / "stage2_params.json").write_text(json.dumps(rep, indent=2))
    log(f"STAGE-2 CV macro F0.5 = {best_f:.5f} (tau {best_tau}, post {best_post}) vs stage 1 "
        f"{max(r1)[0]:.5f}; done in {(time.time() - t0) / 60:.1f} min")


def predict(args):
    from src.predict import check_outputs
    mdir, ndir, out = Path(args.model_dir), Path(args.norm_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prm = json.loads((mdir / "stage2_params.json").read_text())
    tau = args.tau if args.tau is not None else prm["tau"]
    post = args.post if args.post is not None else prm["post"]
    scored = pl.read_parquet(mdir / "test_scored.parquet")
    log(f"test scored pairs: {scored.height:,}; building stage-2 features")
    s1, pool = _load_split(ndir, "test", mdir)
    X = group_features(scored.select(G, "cand_id", "p"), records(s1), records(pool))
    m = lgb.Booster(model_file=str(mdir / "stage2_model.txt"))
    st2 = X.select(G, "cand_id").with_columns(pl.Series("p", m.predict(X.select(S2_FEATURES).to_numpy())))
    st2 = st2.join(scored.select(G, "cand_id", "country"), on=[G, "cand_id"])
    if post != "none":
        st2 = apply_post(st2, numbers_frame(s1, G, "nums1"), numbers_frame(pool, "cand_id", "nums2"), post)
    log(f"decision: threshold {tau}, post {post}")
    s1_ids = s1["entity_id"]
    match = to_lists(decide(st2, {"rule": "threshold", "tau": tau}), s1_ids.to_list())
    cands = pl.read_parquet(mdir / "test_cands.parquet")
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
             .group_by("country").agg(pl.col("ids").list.len().mean().round(3).alias("pred_per_s1"),
                                      (pl.col("ids").list.len() == 0).mean().round(4).alias("empty")))
    log(f"predicted {n.sum():,} matches; {(n == 0).mean():.3%} S1 with no match")
    print(stats)
    if args.validator:
        import subprocess
        from src.config import DATA_DIR
        r = subprocess.run([sys.executable, args.validator, "--matching", str(out / "matching_results.tsv"),
                            "--candidate", str(out / "candidate_pairs.tsv"),
                            "--test-dir", str(Path(args.data_dir or DATA_DIR) / "test")],
                           capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-500:])
    log(f"done -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit", "predict"])
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--n-valid", type=int, default=100_000)
    ap.add_argument("--out-dir", default=str(Path(WORK_DIR) / "output_stage2"))
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--post", default=None)
    ap.add_argument("--validator", default=None)
    args = ap.parse_args()
    fit(args) if args.cmd == "fit" else predict(args)


if __name__ == "__main__":
    main()
