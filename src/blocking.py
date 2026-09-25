"""Candidate generation (blocking) with a multi-key inverted index.

Every record emits hashed blocking keys of six types; an S1 record and an
S2/S3 record become a candidate pair when they share at least one key within
the same country label. Keys (after normalisation + aliases):

  N  single name token                     catches most names
  P  pair of name tokens                   reordered names, empty addresses
  A  name token x address word             generic names ("Summit Inc") + locality
  U  house number x address word           made-up/Indic names with the same address
  W  pair of address words                 addresses without numbers
  C  concatenated name                     handles and domains ("@alikadavis")

Keys whose block in the S2/S3 pool is larger than a cap are dropped (they
carry no signal and explode the pair count). Each shared key adds
``type_weight / log2(1 + block_size)`` to a pair's blocking score; the
``top_k`` best-scoring candidates per S1 record *per source* (S2, S3) are kept.
That final set is exactly what the matching model scores, i.e. it is what is
written to candidate_pairs.tsv.
"""
import time

import polars as pl

from src.normalize import ADDRESS_CANON, NAME_STOPWORDS

STREET_WORDS = sorted(set(ADDRESS_CANON.values()) | set(ADDRESS_CANON.keys()))
# type id: (name, max block size in the pool, weight)
KEY_TYPES = {0: ("N", 200, 1.0), 1: ("P", 300, 1.5), 2: ("A", 300, 2.0),
             3: ("U", 300, 2.0), 4: ("W", 150, 1.0), 5: ("C", 100, 2.0)}
MAX_NAME_TOKENS, MAX_ADDR_WORDS, MAX_NUMBERS = 4, 6, 3


def record_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Blocking keys for records with an ``idx`` column -> (idx, key:u64, t:u8)."""
    b = df.select(
        "idx",
        pl.col("name_core_tokens").list.eval(pl.element().filter(
            (pl.element().str.len_chars() >= 2) & ~pl.element().is_in(NAME_STOPWORDS)))
        .list.head(MAX_NAME_TOKENS).alias("nt"),
        pl.col("addr_tokens").list.eval(pl.element().filter(
            (pl.element().str.len_chars() >= 3) & ~pl.element().is_in(STREET_WORDS)))
        .list.head(MAX_ADDR_WORDS).alias("aw"),
        pl.col("addr_numbers").list.head(MAX_NUMBERS).alias("nu"),
        "name_concat",
    )
    nt = b.select("idx", "nt").explode("nt").drop_nulls("nt")
    aw = b.select("idx", "aw").explode("aw").drop_nulls("aw")
    nu = b.select("idx", "nu").explode("nu").drop_nulls("nu")

    def keys(frame, parts, t):
        return frame.select("idx", pl.concat_str([pl.lit(KEY_TYPES[t][0] + "|")] + parts,
                                                 separator="").alias("k"),
                            pl.lit(t, pl.UInt8).alias("t"))

    nn = nt.join(nt, on="idx", suffix="2").filter(pl.col("nt") < pl.col("nt2"))
    ww = aw.join(aw, on="idx", suffix="2").filter(pl.col("aw") < pl.col("aw2"))
    sep = pl.lit("|")
    frames = [
        keys(nt, [pl.col("nt")], 0),
        keys(nn, [pl.col("nt"), sep, pl.col("nt2")], 1),
        keys(nt.join(aw, on="idx"), [pl.col("nt"), sep, pl.col("aw")], 2),
        keys(nu.join(aw, on="idx"), [pl.col("nu"), sep, pl.col("aw")], 3),
        keys(ww, [pl.col("aw"), sep, pl.col("aw2")], 4),
        keys(b.filter(pl.col("name_concat").str.len_chars() >= 4), [pl.col("name_concat")], 5),
    ]
    return pl.concat(frames).select("idx", pl.col("k").hash(seed=11).alias("key"), "t")


def _keys_chunked(df: pl.DataFrame, chunk: int = 1_000_000) -> pl.DataFrame:
    return pl.concat([record_keys(df.slice(i, chunk)) for i in range(0, df.height, chunk)])


def generate_candidates(queries: pl.DataFrame, pool: pl.DataFrame, top_k: int = 10,
                        chunk_size: int = 25_000, log=print) -> pl.DataFrame:
    """Candidate pairs for every query (S1) record against the S2/S3 pool.

    Returns columns: source1_entity_id, cand_id, source, bscore, n_keytypes.
    Pairs are only formed within the same ``country`` label (checked in EDA:
    0 of 7.6M training matches cross countries).
    """
    type_frame = pl.DataFrame({"t": list(KEY_TYPES), "cap": [v[1] for v in KEY_TYPES.values()],
                               "tw": [v[2] for v in KEY_TYPES.values()]},
                              schema={"t": pl.UInt8, "cap": pl.Int64, "tw": pl.Float64})
    out = []
    for country in sorted(queries["country"].unique().to_list()):
        t0 = time.time()
        q = queries.filter(pl.col("country") == country).with_row_index("idx")
        p = pool.filter(pl.col("country") == country).with_row_index("idx")
        if p.height == 0:
            log(f"  [{country}] no pool records")
            continue
        pk = _keys_chunked(p)
        stats = (pk.group_by("key").agg(pl.len().alias("df"), pl.col("t").first())
                 .join(type_frame, on="t")
                 .filter(pl.col("df") <= pl.col("cap"))
                 .select("key", (pl.col("tw") / (pl.col("df") + 1).log(2)).cast(pl.Float32).alias("w")))
        pk = pk.join(stats, on="key").select(pl.col("idx").alias("pidx"), "key", "w")
        qk = _keys_chunked(q)
        p_meta = p.select(pl.col("idx").alias("pidx"), pl.col("entity_id").alias("cand_id"),
                          pl.col("entity_id").str.slice(0, 2).alias("source"))
        q_meta = q.select("idx", pl.col("entity_id").alias("source1_entity_id"))
        n_pairs = 0
        for start in range(0, q.height, chunk_size):
            qc = qk.filter(pl.col("idx").is_between(start, start + chunk_size - 1))
            pairs = (qc.join(pk, on="key")
                     .group_by("idx", "pidx")
                     .agg(pl.col("w").sum().alias("bscore"),
                          pl.col("t").n_unique().cast(pl.UInt8).alias("n_keytypes"))
                     .join(p_meta, on="pidx")
                     .sort(["idx", "source", "bscore"], descending=[False, False, True])
                     .group_by(["idx", "source"], maintain_order=True).head(top_k)
                     .join(q_meta, on="idx")
                     .select("source1_entity_id", "cand_id", "source", "bscore", "n_keytypes"))
            n_pairs += pairs.height
            out.append(pairs)
        log(f"  [{country}] {q.height:,} queries x {p.height:,} pool -> {n_pairs:,} pairs "
            f"({n_pairs / max(q.height, 1):.1f}/query) in {time.time() - t0:.0f}s")
    if not out:
        return pl.DataFrame(schema={"source1_entity_id": pl.Utf8, "cand_id": pl.Utf8,
                                    "source": pl.Utf8, "bscore": pl.Float32,
                                    "n_keytypes": pl.UInt8})
    return pl.concat(out)


def blocking_recall(cands: pl.DataFrame, gt_pairs: pl.DataFrame, query_ids) -> dict:
    """Share of true (S1, S2/S3) pairs of the given queries present in ``cands``."""
    truth = gt_pairs.filter(pl.col("source1_entity_id").is_in(query_ids))
    found = truth.join(cands.select("source1_entity_id", pl.col("cand_id").alias("matched_id")),
                       on=["source1_entity_id", "matched_id"], how="semi").height
    q = len(set(query_ids)) if not isinstance(query_ids, pl.Series) else query_ids.n_unique()
    return {"true_pairs": truth.height, "found": found,
            "recall": round(found / max(truth.height, 1), 4),
            "cands_per_query": round(cands.height / max(q, 1), 2)}
