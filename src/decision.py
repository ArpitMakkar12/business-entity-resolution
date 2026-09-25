"""Turn pair probabilities into per-entity match lists, and score them.

Steps (all vectorised polars):
  1. Exclusivity: every S2/S3 record belongs to at most one S1 entity (holds
     exactly in the training ground truth), so a candidate is only kept for
     the S1 record where it has the highest probability.
  2. Set choice per S1 entity. The metric is F0.5 per entity; for the top-k
     candidates by probability we estimate
         E[F0.5 | predict top k] ~= 1.25 * sum_{i<=k} p_i / (0.25 * alpha * T + k)
     where T = sum of all candidate probabilities (expected number of true
     matches, alpha corrects for matches outside the candidate set), and
         E[F0.5 | predict nothing] = prod_i (1 - p_i)   (the singleton case).
     The k with the highest expectation is chosen; candidates below ``min_p``
     are never predicted.
  3. ``f05_macro`` implements the official metric (macro F0.5 per S1 entity,
     singletons score 1 only for an empty prediction).
"""
import itertools

import polars as pl

G = "source1_entity_id"


def apply_exclusivity(scored: pl.DataFrame) -> pl.DataFrame:
    """Keep each candidate only for the S1 record where its probability is highest."""
    return scored.filter(pl.col("p") == pl.col("p").max().over("cand_id")).unique(
        ["cand_id"], keep="first", maintain_order=True)


def choose_matches(scored: pl.DataFrame, min_p: float = 0.3, alpha: float = 1.0,
                   exclusive: bool = True) -> pl.DataFrame:
    """Return (source1_entity_id, cand_id) rows selected by the expected-F0.5 rule."""
    d = apply_exclusivity(scored) if exclusive else scored
    d = d.sort([G, "p"], descending=[False, True]).with_columns(
        pl.col("p").cum_sum().over(G).alias("cs"),
        pl.int_range(1, pl.len() + 1).over(G).cast(pl.Float64).alias("k"),
        pl.col("p").sum().over(G).alias("T"),
        (1.0 - pl.col("p")).clip(lower_bound=1e-9).log().sum().over(G).exp().alias("e0"),
    ).with_columns(
        (1.25 * pl.col("cs") / (0.25 * alpha * pl.col("T") + pl.col("k"))).alias("ek"))
    best = (d.filter(pl.col("p") >= min_p).group_by(G)
            .agg(pl.col("ek").max().alias("best_ek"),
                 pl.col("k").sort_by("ek", descending=True).first().alias("best_k"),
                 pl.col("e0").first()))
    return (d.join(best, on=G).filter((pl.col("best_ek") > pl.col("e0"))
                                      & (pl.col("k") <= pl.col("best_k")))
            .select(G, "cand_id"))


def choose_by_threshold(scored: pl.DataFrame, tau: float, exclusive: bool = True):
    d = apply_exclusivity(scored) if exclusive else scored
    return d.filter(pl.col("p") >= tau).select(G, "cand_id")


def to_lists(selected: pl.DataFrame, all_s1_ids) -> pl.DataFrame:
    """One row per S1 id with the (possibly empty) list of chosen ids."""
    lists = selected.group_by(G).agg(pl.col("cand_id").alias("ids"))
    base = pl.DataFrame({G: pl.Series(all_s1_ids, dtype=pl.Utf8)})
    return base.join(lists, on=G, how="left").with_columns(
        pl.col("ids").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))))


def f05_per_entity(pred: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """pred: (G, ids list); truth: (G, matched list) -> per-entity F0.5 score."""
    d = truth.select(G, "matched").join(pred, on=G, how="left").with_columns(
        pl.col("ids").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))))
    tp = pl.col("ids").list.set_intersection(pl.col("matched")).list.len().cast(pl.Float64)
    n_pred = pl.col("ids").list.len().cast(pl.Float64)
    n_true = pl.col("matched").list.len().cast(pl.Float64)
    prec = tp / n_pred.clip(lower_bound=1)
    rec = tp / n_true.clip(lower_bound=1)
    f = pl.when((n_true == 0) & (n_pred == 0)).then(1.0) \
        .when((n_true == 0) | (n_pred == 0) | (tp == 0)).then(0.0) \
        .otherwise(1.25 * prec * rec / (0.25 * prec + rec))
    return d.with_columns(f.alias("f05"), prec.alias("precision"), rec.alias("recall"),
                          (n_true == 0).alias("is_singleton"))


def f05_macro(pred: pl.DataFrame, truth: pl.DataFrame) -> float:
    return float(f05_per_entity(pred, truth)["f05"].mean())


def tune_decision(scored: pl.DataFrame, truth: pl.DataFrame, log=print) -> dict:
    """Grid-search the decision rule on validation data; return the best params."""
    ids = truth[G].to_list()
    results = []
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        f = f05_macro(to_lists(choose_by_threshold(scored, tau), ids), truth)
        results.append(({"rule": "threshold", "tau": tau}, f))
    for min_p, alpha in itertools.product((0.1, 0.2, 0.3, 0.4, 0.5), (0.8, 1.0, 1.25, 1.5)):
        f = f05_macro(to_lists(choose_matches(scored, min_p, alpha), ids), truth)
        results.append(({"rule": "expected_f", "min_p": min_p, "alpha": alpha}, f))
    results.sort(key=lambda r: -r[1])
    for params, f in results[:5]:
        log(f"  {f:.5f}  {params}")
    best, f = results[0]
    return {**best, "valid_f05": round(f, 5)}


def decide(scored: pl.DataFrame, params: dict) -> pl.DataFrame:
    if params["rule"] == "threshold":
        return choose_by_threshold(scored, params["tau"])
    return choose_matches(scored, params["min_p"], params["alpha"])
