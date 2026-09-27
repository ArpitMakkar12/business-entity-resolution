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
