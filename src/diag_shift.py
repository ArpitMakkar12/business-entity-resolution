"""Where do test predictions differ from validation predictions? (adversarial check)

Leaderboard probes imply that India test entities score far below validation
(roughly 0.93 vs 0.97) while US test is close to validation. If the model makes
a test-only kind of mistake, the predicted pairs behind it must look different
from anything the model predicts on validation.

For one country this script:
  1. takes the PREDICTED pairs on validation (model, p90 rule, threshold) with
     their 61 features and labels,
  2. recomputes the same features for the predicted pairs of a sample of test
     S1 entities (full candidate lists, so the group features are exact),
  3. trains a LightGBM "adversary" to tell test predictions from validation
     predictions (3-fold CV AUC; 0.5 = no difference),
  4. prints the features that separate them, legal-form conflict rates, and
     examples of the most "test-only" predicted pairs, plus validation false
     positives for comparison.

Usage: python -m src.diag_shift --country India [--model-dir WORK_DIR/model_v3] [--n-s1 60000]
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.aliases import apply_aliases, load_aliases
from src.config import SEED, WORK_DIR
from src.decision import G, decide
from src.features import FEATURES, add_name_freq, compute_features_chunked, token_idf
from src.postprocess import apply_post, numbers_frame
from src.train import load_normalized, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--model-dir", default=str(Path(WORK_DIR) / "model_v3"))
    ap.add_argument("--country", default="India")
    ap.add_argument("--n-s1", type=int, default=60_000)
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--n-valid", type=int, default=100_000)
    ap.add_argument("--out", default=None, help="write the top test-only pairs here (TSV)")
    ap.add_argument("--exclude-size", action="store_true",
                    help="adversary ignores features that depend on table sizes (idf, name freq, bscore)")
    args = ap.parse_args()
    mdir, ndir, c = Path(args.model_dir), Path(args.norm_dir), args.country
    tau = json.loads((mdir / "params.json").read_text())["decision"].get("tau", 0.7)
    model = lgb.Booster(model_file=str(mdir / "model.txt"))
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_fmt_str_lengths(36)

    # ---------------- validation predicted pairs --------------------------------
    ids = (pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id"])
           .sample(fraction=1.0, shuffle=True, seed=SEED)["entity_id"])
    n_tr = min(args.n_train, int(len(ids) * 0.8))
    valid_ids = ids[n_tr:n_tr + min(args.n_valid, len(ids) - n_tr)]
    vf = (pl.scan_parquet(mdir / "train_feats.parquet")
          .filter(pl.col(G).is_in(valid_ids.implode()) & (pl.col("country") == c)).collect())
    if vf.height == 0:      # no training data for this country (France): compare with all validation
        log(f"no validation data for {c}; using all validation countries")
        vf = pl.scan_parquet(mdir / "train_feats.parquet").filter(pl.col(G).is_in(valid_ids.implode())).collect()
    vf = vf.with_columns(pl.Series("p", model.predict(vf.select(FEATURES).to_numpy())))
    tr_s1 = pl.read_parquet(ndir / "train_s1.parquet", columns=["entity_id", "addr_numbers", "name_legal",
                                                                 "business_name", "business_address"])
    tr_pool = pl.concat([pl.read_parquet(ndir / f"train_s{k}.parquet",
                                         columns=["entity_id", "addr_numbers", "name_legal", "business_name",
                                                  "business_address"]) for k in (2, 3)])
    if (mdir / "siblings.parquet").exists():
        sib = pl.read_parquet(mdir / "siblings.parquet").select(tr_pool.columns)
        tr_pool = pl.concat([tr_pool, sib])
    vs = apply_post(vf.select(G, "cand_id", "country", "p"), numbers_frame(tr_s1, G, "nums1"),
                    numbers_frame(tr_pool, "cand_id", "nums2"), "conflict_p90")
    vsel = decide(vs, {"rule": "threshold", "tau": tau})
    val = vf.join(vsel, on=[G, "cand_id"]).with_columns(pl.lit(0).alias("is_test"))
    log(f"validation {c}: {vf[G].n_unique():,} S1, {val.height:,} predicted pairs, "
        f"precision {val['y'].mean():.4f}")

    # ---------------- test predicted pairs (sample of S1, exact features) -------
    s1, pool = load_normalized(ndir, "test")
    al = load_aliases(mdir / "aliases.json")
    s1, pool = apply_aliases(s1, al), apply_aliases(pool, al)
    s1, pool = add_name_freq(s1), add_name_freq(pool)
    sc = pl.read_parquet(mdir / "test_scored.parquet")
    sc = apply_post(sc, numbers_frame(s1, G, "nums1"), numbers_frame(pool, "cand_id", "nums2"), "conflict_p90")
    tsel = decide(sc, {"rule": "threshold", "tau": tau}).join(sc.select(G, "cand_id", "p"), on=[G, "cand_id"])
    samp = (s1.filter(pl.col("country") == c).select("entity_id")
            .sample(n=min(args.n_s1, s1.filter(pl.col("country") == c).height), seed=SEED)["entity_id"])
    cands = pl.read_parquet(mdir / "test_cands.parquet").filter(pl.col(G).is_in(samp.implode()))
    idf = token_idf(s1, pool)
    tf = pl.concat(list(compute_features_chunked(cands, s1, pool, idf, log=log)))
    test = tf.join(tsel, on=[G, "cand_id"]).with_columns(pl.lit(1).alias("is_test"), pl.lit(-1).alias("y"))
    log(f"test {c}: {len(samp):,} sampled S1, {test.height:,} predicted pairs")

    # ---------------- legal-form conflicts ---------------------------------------
    def legal(df, s1t, poolt):
        d = (df.join(s1t.select(pl.col("entity_id").alias(G), pl.col("name_legal").alias("l1")), on=G, how="left")
             .join(poolt.select(pl.col("entity_id").alias("cand_id"), pl.col("name_legal").alias("l2")),
                   on="cand_id", how="left"))
        both = (pl.col("l1").fill_null("") != "") & (pl.col("l2").fill_null("") != "")
        return d.with_columns((both & (pl.col("l1") != pl.col("l2"))).alias("legal_conflict"))
    val = legal(val, tr_s1, tr_pool)
    test = legal(test, s1, pool)
    print(f"\n=== {c}: legal-form conflict among predicted pairs ===")
    print(f"  validation: {val['legal_conflict'].mean():.4f} of predicted pairs "
          f"(precision of those {val.filter(pl.col('legal_conflict'))['y'].mean():.3f}); "
          f"test: {test['legal_conflict'].mean():.4f}")
    ent_v = val.group_by(G).agg(pl.col("legal_conflict").all().alias("all_conf"), pl.col("y").max().alias("any_true"))
    ent_t = test.group_by(G).agg(pl.col("legal_conflict").all().alias("all_conf"))
    print(f"  entities whose EVERY predicted pair has a legal conflict: validation "
          f"{ent_v['all_conf'].mean():.4f} (of them truly matched: "
          f"{ent_v.filter(pl.col('all_conf'))['any_true'].mean():.3f}), test {ent_t['all_conf'].mean():.4f}")

    # ---------------- feature shift ----------------------------------------------
    print(f"\n=== {c}: mean feature value of predicted pairs, validation vs test (largest differences) ===")
    rows = []
    for f in FEATURES:
        a, b = val[f].cast(pl.Float64), test[f].cast(pl.Float64)
        sd = float(pl.concat([a, b]).std() or 1.0) or 1.0
        rows.append((f, float(a.mean()), float(b.mean()), (float(b.mean()) - float(a.mean())) / sd))
    sh = pl.DataFrame(rows, schema=["feature", "valid_mean", "test_mean", "shift_sd"], orient="row")
    print(sh.with_columns(pl.col("shift_sd").abs().alias("_a")).sort("_a", descending=True).drop("_a").head(15)
          .with_columns(pl.col(pl.Float64).round(4)))

    # ---------------- adversarial classifier -------------------------------------
    n = min(val.height, test.height)
    def norm(df):
        return df.select(G, "cand_id", pl.col("is_test").cast(pl.Int32), pl.col("p").cast(pl.Float64),
                         pl.col("y").cast(pl.Int32), *[pl.col(f).cast(pl.Float32) for f in FEATURES])
    both = pl.concat([norm(val.sample(n=n, seed=1)), norm(test.sample(n=n, seed=1))])
    SIZE = {"s1_name_freq", "cand_name_freq", "addr_empty_amb", "bscore", "bscore_gap"}
    ADV = [f for f in FEATURES if not (args.exclude_size and (f.startswith("idf_") or f in SIZE))]
    X, y = both.select(ADV).to_numpy(), both["is_test"].to_numpy()
    fold = (both[G].hash(seed=5) % 3).to_numpy()
    oof = np.zeros(len(y))
    imp = np.zeros(len(ADV))
    prm = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
           "feature_fraction": 0.8, "verbose": -1, "seed": SEED}
    for k in range(3):
        m = lgb.train(prm, lgb.Dataset(X[fold != k], y[fold != k]), 300)
        oof[fold == k] = m.predict(X[fold == k])
        imp += m.feature_importance("gain")
    ranks = np.argsort(np.argsort(oof)).astype(np.float64)       # ascending ranks 0..n-1
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    auc = (ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    print(f"\n=== {c}: adversary test-vs-validation predicted pairs: AUC {auc:.4f} (0.5 = same) ===")
    top = sorted(zip(ADV, imp), key=lambda t: -t[1])[:12]
    print("  separating features: " + ", ".join(f"{f}={v:.0f}" for f, v in top))
    both = both.with_columns(pl.Series("adv", oof))
    for thr in (0.8, 0.9, 0.95):
        t_share = both.filter(pl.col("is_test") == 1)["adv"].gt(thr).mean()
        v = both.filter((pl.col("is_test") == 0) & (pl.col("adv") > thr))
        print(f"  adv > {thr}: {t_share:.4f} of test predicted pairs vs "
              f"{both.filter(pl.col('is_test') == 0)['adv'].gt(thr).mean():.4f} of validation "
              f"(validation precision there {v['y'].mean() if v.height else float('nan'):.3f}, n={v.height})")

    names1 = s1.select(pl.col("entity_id").alias(G), pl.col("business_name").alias("s1_name"),
                       pl.col("business_address").alias("s1_addr"))
    names2 = pool.select(pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("rec_name"),
                         pl.col("business_address").alias("rec_addr"))
    ex = (both.filter(pl.col("is_test") == 1).sort("adv", descending=True).head(400)
          .join(names1, on=G).join(names2, on="cand_id"))
    print(f"\n=== {c}: MOST TEST-ONLY predicted pairs (random 30 of the top 400 by adversary score) ===")
    print(ex.sample(n=min(30, ex.height), seed=2).select(
        "s1_name", "s1_addr", "rec_name", "rec_addr", pl.col("p").round(3), pl.col("adv").round(3),
        pl.col("name_tset").round(2), pl.col("addr_tset").round(2), "num_first_match"))
    if args.out:
        ex.select("s1_name", "s1_addr", "rec_name", "rec_addr", "p", "adv", "name_tset", "addr_tset",
                  "num_first_match").write_csv(args.out, separator="\t")
        log(f"top test-only pairs -> {args.out}")

    fp = (val.filter(pl.col("y") == 0).sample(n=min(15, val.filter(pl.col("y") == 0).height), seed=3)
          .join(tr_s1.select(pl.col("entity_id").alias(G), pl.col("business_name").alias("s1_name"),
                             pl.col("business_address").alias("s1_addr")), on=G)
          .join(tr_pool.select(pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("rec_name"),
                               pl.col("business_address").alias("rec_addr")), on="cand_id"))
    print(f"\n=== {c}: validation FALSE POSITIVES for comparison (random 15) ===")
    print(fp.select("s1_name", "s1_addr", "rec_name", "rec_addr", pl.col("p").round(3)))


if __name__ == "__main__":
    main()
