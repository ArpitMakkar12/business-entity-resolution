"""Who gets an S2/S3 record that is likely (p >= tau) for several S1 entities?

Default (in src.decision): the S1 with the highest probability. On test this
often decides between near-identical probabilities of look-alike S1 entities
(France: "Calais Ecole SARL, 62 Rue Becquerel" vs "Calais Ecole SARL, 62 Rue
Arago"; "Lille Ecole SARL" vs "Lille Comite SARL" in the same building), which
splits one business's copies between two S1 records. Validation never shows
this because only a sample of training S1 records is scored.

Among the claimants within ``margin`` of the best probability:
  strength  the S1 whose whole predicted set is strongest (sum of p)
  explain   the S1 that explains the record best: shares a house number first,
            then the highest name + address similarity (core name and cleaned
            address, mean of token-set and token-sort ratio) with +-10 for an
            agreeing / conflicting legal form, then probability
Claimants further than ``margin`` below the best probability never win.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from src.config import N_JOBS

G = "source1_entity_id"


def reassign(scored: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, tau: float,
             mode: str, margin: float, log=print, examples_path=None) -> pl.DataFrame:
    """Return the pairs with p >= tau, each record kept for exactly one S1."""
    hi = scored.filter(pl.col("p") >= tau)
    base_winner = (hi.sort(["cand_id", "p"], descending=[False, True])
                   .unique("cand_id", keep="first", maintain_order=True).select("cand_id", pl.col(G).alias("_argmax")))
    elig = hi.filter(pl.col("p") >= pl.col("p").max().over("cand_id") - margin)
    multi = elig.filter(pl.len().over("cand_id") >= 2)
    single = elig.filter(pl.len().over("cand_id") == 1)
    log(f"assignment '{mode}' (margin {margin}): {multi['cand_id'].n_unique():,} contested records, "
        f"{multi.height:,} claims")
    if multi.height == 0:
        return single
    if mode == "strength":
        st = hi.group_by(G).agg(pl.col("p").sum().alias("_key"))
        multi = multi.join(st, on=G).join(base_winner, on="cand_id").with_columns(
            (pl.col(G) == pl.col("_argmax")).cast(pl.Int8).alias("_isbest")).drop("_argmax")
        keys = ["_key", "_isbest", "p"]
    else:
        d = (multi.join(s1.select(pl.col("entity_id").alias(G), pl.col("name_core").alias("n1"),
                                  pl.col("addr_clean").alias("a1"), pl.col("addr_numbers").alias("nu1"),
                                  pl.col("name_legal").alias("l1")), on=G)
             .join(pool.select(pl.col("entity_id").alias("cand_id"), pl.col("name_core").alias("n2"),
                               pl.col("addr_clean").alias("a2"), pl.col("addr_numbers").alias("nu2"),
                               pl.col("name_legal").alias("l2")), on="cand_id"))
        def sim(a, b):
            # token-set alone gives 100 to any subset ("christ cohen" vs "cohen christ
            # pediatric"), token-sort penalises the missing words: use their mean
            a, b = d[a].fill_null("").to_list(), d[b].fill_null("").to_list()
            return (cpdist(a, b, scorer=fuzz.token_set_ratio, workers=N_JOBS, dtype=np.float32)
                    + cpdist(a, b, scorer=fuzz.token_sort_ratio, workers=N_JOBS, dtype=np.float32)) / 2
        ns, ad = sim("n1", "n2"), sim("a1", "a2")
        multi = d.with_columns(
            (pl.col("nu1").list.set_intersection(pl.col("nu2")).list.len() > 0).fill_null(False)
            .cast(pl.Int8).alias("_num"),
            # legal form: +10 when both carry the same one, -10 when both carry different
            # ones (sibling businesses often differ only in it: "Calais Club" / "Calais Club SA")
            (pl.Series("_sim", np.round(ns + ad, 1))
             + pl.when((pl.col("l1").fill_null("") == "") | (pl.col("l2").fill_null("") == "")).then(0.0)
             .when(pl.col("l1") == pl.col("l2")).then(10.0).otherwise(-10.0)).alias("_sim")
        ).drop("n1", "n2", "a1", "a2", "nu1", "nu2", "l1", "l2")
        # only a clear similarity advantage (>= 10 points) overrides the probability
        multi = multi.with_columns((pl.col("_sim") / 10).floor().alias("_sim"))
        multi = multi.join(base_winner, on="cand_id").with_columns(
            (pl.col(G) == pl.col("_argmax")).cast(pl.Int8).alias("_isbest")).drop("_argmax")
        keys = ["_num", "_sim", "_isbest", "p"]
    win = (multi.sort(["cand_id"] + keys, descending=[False] + [True] * len(keys))
           .unique("cand_id", keep="first", maintain_order=True))
    changed = win.join(base_winner, on="cand_id").filter(pl.col(G) != pl.col("_argmax"))
    log(f"  {changed.height:,} records change owner vs highest-probability assignment "
        f"({changed[G].n_unique():,} gaining S1, {changed['_argmax'].n_unique():,} losing S1)")
    if "country" in changed.columns:
        log("  by country: " + str(dict(changed.group_by("country").len().iter_rows())))
    if examples_path is not None and changed.height:
        names = s1.select(pl.col("entity_id"), "business_name", "business_address")
        old_p = hi.select(pl.col(G).alias("_argmax"), "cand_id", pl.col("p").alias("old_p"))
        ex = (changed.sample(n=min(300, changed.height), seed=1).join(old_p, on=["_argmax", "cand_id"])
              .join(names.rename({"entity_id": G, "business_name": "new_owner_name",
                                  "business_address": "new_owner_addr"}), on=G)
              .join(names.rename({"entity_id": "_argmax", "business_name": "old_owner_name",
                                  "business_address": "old_owner_addr"}), on="_argmax")
              .join(pool.select(pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("record_name"),
                                pl.col("business_address").alias("record_addr")), on="cand_id")
              .select("country", "record_name", "record_addr", "new_owner_name", "new_owner_addr",
                      pl.col("p").round(4).alias("new_p"), "old_owner_name", "old_owner_addr",
                      pl.col("old_p").round(4)))
        ex.write_csv(examples_path, separator="\t")
        log(f"  examples of changed owners -> {examples_path}")
    cols = scored.columns
    return pl.concat([single.select(cols), win.select(cols)])


def joint_probability(scored: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, tau: float,
                      log=print, examples_path=None, p_floor: float = 0.05) -> pl.DataFrame:
    """Exclusivity-aware probabilities.

    The pair model scores every (S1, record) pair on its own, so a record that
    fits two S1 entities well gets e.g. p = 0.99 for both, although at most one
    of them can own it (exclusivity holds exactly in the training data). If the
    pair probabilities are treated as independent evidence and exactly zero or
    one claimant owns the record, the probability that S1 i owns it is

        q_i = o_i / (1 + sum_j o_j),   o = p / (1 - p)   (odds)

    For a single claimant q = p, so everything validated so far is unchanged.
    Two near-certain claimants (0.99 / 0.98) give q = 0.66 / 0.33, and for the
    F0.5 metric it is better not to predict such a coin flip at all; a clear
    winner (0.99 vs 0.75) keeps q = 0.96. Validation cannot show this effect
    because only a sample of training S1 records competes there.
    """
    eps = 1e-6
    d = scored.with_columns(pl.col("p").clip(eps, 1 - eps).alias("_pc"))
    d = d.with_columns(pl.when(pl.col("_pc") >= p_floor)
                       .then(pl.col("_pc") / (1 - pl.col("_pc"))).otherwise(0.0).alias("_odds"))
    d = d.with_columns((pl.col("_odds") / (1 + pl.col("_odds").sum().over("cand_id"))).alias("_q"),
                       pl.len().over("cand_id").alias("_n"))
    d = d.with_columns(pl.when(pl.col("_pc") >= p_floor).then(pl.col("_q")).otherwise(pl.col("p")).alias("_q"))
    before = d.filter((pl.col("p") >= tau) & (pl.col("p") == pl.col("p").max().over("cand_id")))
    dropped = before.filter(pl.col("_q") < tau)
    log(f"joint probability: {dropped.height:,} of {before.height:,} winning pairs fall below tau {tau} "
        f"(records that fit several S1 entities about equally)")
    if "country" in dropped.columns:
        log("  by country: " + str(dict(dropped.group_by("country").len().sort("country").iter_rows())))
    if examples_path is not None and dropped.height:
        rival = (d.filter(pl.col("p") >= tau).sort("p", descending=True)
                 .group_by("cand_id", maintain_order=True).agg(pl.col(G).slice(1, 1).first().alias("_rival"),
                                                               pl.col("p").slice(1, 1).first().alias("rival_p")))
        names = s1.select(pl.col("entity_id"), "business_name", "business_address")
        ex = (dropped.sample(n=min(300, dropped.height), seed=1).join(rival, on="cand_id", how="left")
              .join(names.rename({"entity_id": G, "business_name": "owner_name", "business_address": "owner_addr"}),
                    on=G)
              .join(names.rename({"entity_id": "_rival", "business_name": "rival_name",
                                  "business_address": "rival_addr"}), on="_rival", how="left")
              .join(pool.select(pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("record_name"),
                                pl.col("business_address").alias("record_addr")), on="cand_id")
              .select("country", "record_name", "record_addr", "owner_name", "owner_addr",
                      pl.col("p").round(4).alias("owner_p"), "rival_name", "rival_addr",
                      pl.col("rival_p").round(4), pl.col("_q").round(3).alias("q")))
        ex.write_csv(examples_path, separator="\t")
        log(f"  examples of dropped pairs -> {examples_path}")
    return d.with_columns(pl.col("_q").alias("p")).drop("_pc", "_odds", "_q", "_n")
