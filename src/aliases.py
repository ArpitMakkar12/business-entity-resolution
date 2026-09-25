"""Learn name-token aliases from training ground-truth pairs.

Indic-script names are romanised phonetically ('हेल्थकेयर' -> 'helthkeyar'),
which does not spell the English word the S1 record uses ('healthcare').
Here we learn such mappings *from the provided training data only*:

  1. take ground-truth pairs whose S2/S3 name contained Indic script,
  2. keep pairs whose core names have the same number of tokens
     (transliterated names keep word order), align tokens by position,
  3. keep a mapping src -> dst when it is seen >= ``min_count`` times, is the
     dominant target for src (share >= ``min_share``) and src is not itself a
     common S1 word (so real English words are never rewritten).

The learned table is saved as JSON and applied identically to train and test.
"""
import json
from pathlib import Path

import polars as pl

from src.normalize import NAME_STOPWORDS, _skeleton


def learn_aliases(s1: pl.DataFrame, pool: pl.DataFrame, gt_pairs: pl.DataFrame,
                  min_count: int = 3, min_share: float = 0.5,
                  max_s1_freq: int = 5) -> dict:
    """Return {romanised_token: s1_token} learned from positional alignment."""
    left = s1.select(pl.col("entity_id").alias("source1_entity_id"),
                     pl.col("name_core_tokens").alias("t1"))
    right = (pool.filter(pl.col("name_has_indic"))
             .select(pl.col("entity_id").alias("matched_id"),
                     pl.col("name_core_tokens").alias("t2")))
    pairs = (gt_pairs.join(right, on="matched_id").join(left, on="source1_entity_id")
             .filter((pl.col("t1").list.len() == pl.col("t2").list.len())
                     & (pl.col("t1").list.len() > 0)))
    aligned = (pairs.select(pl.col("t2").alias("src"), pl.col("t1").alias("dst"))
               .explode(["src", "dst"]).filter(pl.col("src") != pl.col("dst")))
    counts = aligned.group_by("src", "dst").len()
    totals = counts.group_by("src").agg(pl.col("len").sum().alias("total"))
    best = (counts.sort("len", descending=True).group_by("src", maintain_order=True).first()
            .join(totals, on="src"))
    s1_freq = (s1.select(pl.col("name_core_tokens").explode().alias("src"))
               .group_by("src").len().rename({"len": "s1_freq"}))
    keep = (best.join(s1_freq, on="src", how="left")
            .with_columns(pl.col("s1_freq").fill_null(0))
            .filter((pl.col("len") >= min_count)
                    & (pl.col("len") / pl.col("total") >= min_share)
                    & (pl.col("s1_freq") < max_s1_freq)))
    return dict(zip(keep["src"].to_list(), keep["dst"].to_list()))


def apply_aliases(df: pl.DataFrame, aliases: dict) -> pl.DataFrame:
    """Rewrite name tokens with the alias table and rebuild derived name views."""
    if not aliases:
        return df
    content = ~pl.element().is_in(NAME_STOPWORDS)
    df = df.with_columns(
        pl.col("name_core_tokens").list.eval(pl.element().replace(aliases))
        .list.unique(maintain_order=True).alias("name_core_tokens"))
    return df.with_columns(
        pl.col("name_core_tokens").list.join(" ").alias("name_core"),
        pl.col("name_core_tokens").list.eval(pl.element().filter(content)).list.join("")
        .alias("name_concat"),
        _skeleton(pl.col("name_core_tokens").list.eval(pl.element().filter(content))
                  .list.join(" ")).alias("name_skeleton"),
    )


def save_aliases(aliases: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(aliases, ensure_ascii=False, indent=0, sort_keys=True),
                          encoding="utf-8")


def load_aliases(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
