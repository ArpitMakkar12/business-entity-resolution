"""Synthetic "sibling business" negatives for training.

The test set contains many distractor records that are near-copies of a real
entity: same street, a neighbouring house number and one name word changed,
often with several noisy duplicates of their own. E.g. S1 "Grand Future LLC,
2536 Heathcote Lane" vs distractor "Grand Future Partners, 2547 Heathcote Ln".
The training data rarely contains them, so the model learned that such pairs
are matches.

This module fabricates such siblings *from the training data only*: for a
sample of training S1 entities it takes 1-3 of their true matched S2/S3
records (already realistically noisy) and applies one consistent sibling
identity to all of them:
  * the S1 entity's first house number is shifted by +-1..30 in the copy's
    address (copies whose address does not carry that number are skipped:
    same address + one changed word is ordinary noise in this data, e.g.
    "Rocky Center" is a true match of "Rocky Electric", so it must not be
    labelled a negative),
  * one core name word of the S1 name is replaced by another business word
    from the same country's S1 vocabulary (70%), or a word is added (30%).
The resulting records get new ids (S2-syn*/S3-syn*), are not in the ground
truth, and are therefore negatives for every S1 entity. They are added to the
training pool before blocking, so they compete as candidates exactly like the
test distractors do.
"""
import random
import re

import polars as pl

from src.normalize import LEGAL_CANON

G = "source1_entity_id"
_WORD = re.compile(r"[A-Za-z]{3,}")
_LEGAL = {k.lower() for k in LEGAL_CANON} | {"and", "the", "of"}


def _vocab(s1: pl.DataFrame, top: int = 3000) -> dict:
    """Frequent alphabetic name words per country from (clean) S1 names."""
    words = (s1.select("country", pl.col("business_name").str.extract_all(r"[A-Za-z]{3,}").alias("w"))
             .explode("w").drop_nulls("w")
             .filter(~pl.col("w").str.to_lowercase().is_in(list(_LEGAL)))
             .group_by("country", "w").len().sort("len", descending=True))
    out = {}
    for key, g in words.group_by("country"):
        country = key[0] if isinstance(key, tuple) else key
        out[country] = g.sort("len", descending=True).head(top)["w"].to_list()
    return out


def _shift_number(addr: str, num: str, delta: int) -> str:
    """Replace the house number ``num`` (with optional leading zeros) by num+delta."""
    if not addr or not num:
        return addr
    new = str(max(1, int(num) + delta))
    return re.sub(rf"(?<!\d)0*{num}(?!\d)", new, addr, count=1)


def _swap_word(name: str, target: str, new_word: str, mode: str) -> str:
    """Replace ``target`` (case-insensitive) by ``new_word``; append if absent or mode=add."""
    if not name:
        return name
    if mode == "replace" and target:
        pat = re.compile(rf"\b{re.escape(target)}\b", re.IGNORECASE)
        if pat.search(name):
            def repl(m):
                w = m.group(0)
                return new_word.upper() if w.isupper() else new_word.lower() if w.islower() else new_word
            return pat.sub(repl, name, count=1)
    return f"{name} {new_word.upper() if name.isupper() else new_word}"


def make_siblings(query: pl.DataFrame, pool: pl.DataFrame, gt_pairs: pl.DataFrame,
                  frac: float = 0.4, seed: int = 7) -> pl.DataFrame:
    """Raw sibling records (entity_id, business_name, business_address, country).

    query: S1 records (entity_id, business_name, business_address, country, addr_numbers)
    pool:  S2/S3 records with the same raw columns
    """
    rng = random.Random(seed)
    vocab = _vocab(query)
    chosen = query.sample(fraction=frac, seed=seed).select(
        pl.col("entity_id").alias(G), pl.col("business_name").alias("s1_name"),
        pl.col("addr_numbers").list.first().alias("s1_num"), "country")
    base = (gt_pairs.join(chosen, on=G)
            .join(pool.select(pl.col("entity_id").alias("matched_id"), "business_name",
                              "business_address"), on="matched_id")
            .sample(fraction=1.0, shuffle=True, seed=seed)
            .with_columns(pl.int_range(pl.len()).over(G).alias("_r")))
    # per S1 sibling identity: number of copies, number shift, word change
    ident = {}
    for sid, s1_name, s1_num, country in chosen.iter_rows():
        words = [w for w in _WORD.findall(s1_name or "") if w.lower() not in _LEGAL]
        pool_words = vocab.get(country) or ["Group"]
        new_word = rng.choice(pool_words)
        while words and new_word.lower() in {w.lower() for w in words}:
            new_word = rng.choice(pool_words)
        delta = rng.choice([-1, 1]) * rng.randint(1, 30)
        ident[sid] = (rng.randint(1, 3), s1_num, delta, rng.choice(words) if words else "",
                      new_word, "replace" if rng.random() < 0.7 else "add")
    out, n = [], 0
    for sid, mid, name, addr, r in base.select(G, "matched_id", "business_name",
                                               "business_address", "_r").iter_rows():
        k, s1_num, delta, target, new_word, mode = ident[sid]
        if r >= k or not name or " " not in name.strip() or re.search(r"[@#]|\.com|www", name):
            continue          # handles / domains do not make convincing siblings
        if not (s1_num and s1_num.isdigit()):
            continue          # a sibling must differ in house number (see module doc)
        new_addr = _shift_number(addr or "", s1_num, delta)
        if new_addr == (addr or ""):
            continue          # copy does not carry the S1 number: identical address would
                              # contradict real noisy matches ("Rocky Center" = "Rocky Electric")
        n += 1
        out.append((f"{mid[:2]}-syn{n}", _swap_word(name or "", target, new_word, mode), new_addr,
                    sid))
    sib = pl.DataFrame(out, schema=["entity_id", "business_name", "business_address", G], orient="row")
    return sib.join(chosen.select(G, "country"), on=G).drop(G)
