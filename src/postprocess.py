"""Post-processing of pair probabilities: house-number conflict rules.

The test set contains many "sibling" distractors that the training data
rarely shows: a near-copy of a business on the same street with a slightly
different house number and one name word changed (e.g. S1 "Grand Future LLC,
2536 Heathcote Ln" vs "Grand Future Partners, 2547 Heathcote Ln"). The model
gives such pairs 0.7-0.95 because in training a house-number mismatch is only
weak evidence. These rules down-weight pairs whose house numbers conflict.

A pair *conflicts* when both records carry numbers and share none.
Variants (probability set to 0 when the rule fires):
  conflict_all     every conflicting pair
  conflict_pXX     conflicting pairs with p < 0.XX
  group            conflicting pairs, only when the same S1 already has a
                   likely match (p >= 0.5) that shares a house number with it
  group_or_p95     group rule, or any conflicting pair with p < 0.95
"""
import polars as pl

G = "source1_entity_id"
VARIANTS = {
    "none": None,
    "conflict_all": {"p_max": 2.0, "group": False},
    "conflict_p99": {"p_max": 0.99, "group": False},
    "conflict_p95": {"p_max": 0.95, "group": False},
    "conflict_p90": {"p_max": 0.90, "group": False},
    "group": {"p_max": -1.0, "group": True},
    "group_or_p95": {"p_max": 0.95, "group": True},
}


def number_flags(scored: pl.DataFrame, s1_nums: pl.DataFrame, pool_nums: pl.DataFrame,
                 p_floor: float = 0.3):
    """Split scored pairs into (flagged pairs with p >= p_floor, untouched rest).

    s1_nums: (source1_entity_id, nums1)   pool_nums: (cand_id, nums2)
    """
    hi = scored.filter(pl.col("p") >= p_floor)
    lo = scored.filter(pl.col("p") < p_floor)
    hi = hi.join(s1_nums, on=G, how="left").join(pool_nums, on="cand_id", how="left")
    inter = pl.col("nums1").list.set_intersection(pl.col("nums2")).list.len()
    has1 = pl.col("nums1").list.len().fill_null(0) > 0
    has2 = pl.col("nums2").list.len().fill_null(0) > 0
    hi = hi.with_columns((has1 & has2 & (inter.fill_null(0) == 0)).alias("conflict"),
                         (inter.fill_null(0) > 0).alias("num_match"))
    hi = hi.with_columns(((pl.col("p") >= 0.5) & pl.col("num_match")).any().over(G)
                         .alias("grp_has_match"))
    return hi, lo


def apply_post(scored: pl.DataFrame, s1_nums: pl.DataFrame, pool_nums: pl.DataFrame,
               variant: str) -> pl.DataFrame:
    """Return ``scored`` with probabilities zeroed where the chosen rule fires."""
    cfg = VARIANTS[variant]
    if cfg is None:
        return scored
    hi, lo = number_flags(scored, s1_nums, pool_nums)
    fire = pl.col("conflict") & (pl.col("p") < cfg["p_max"])
    if cfg["group"]:
        fire = fire | (pl.col("conflict") & pl.col("grp_has_match"))
    hi = hi.with_columns(pl.when(fire).then(0.0).otherwise(pl.col("p")).alias("p"))
    return pl.concat([hi.select(scored.columns), lo.select(scored.columns)])


def numbers_frame(df: pl.DataFrame, key: str, name: str) -> pl.DataFrame:
    """(entity_id, addr_numbers) -> (key, name) for joining."""
    return df.select(pl.col("entity_id").alias(key), pl.col("addr_numbers").alias(name))
