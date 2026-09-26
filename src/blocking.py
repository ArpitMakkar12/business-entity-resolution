"""Candidate generation (blocking): multi-key inverted index + similarity re-rank.

Every record emits hashed blocking keys of eight types; an S1 record and an
S2/S3 record are paired when they share at least one key within the same
country label. Keys (after normalisation + aliases):

  N  single name token                  most names
  P  pair of name tokens                reordered names, empty addresses
  A  name token x address word          generic names ("Summit Inc") + locality
  U  house number x address word        made-up / Indic names at the same address
  W  pair of address words              addresses without numbers
  C  concatenated name                  handles and domains ("@alikadavis")
  B  house number x name token          same number + name, reworded address
  M  pair of house numbers              multi-number Indian addresses ("1206/1207")
  T  triple of name tokens               names built from common words ("Urgent Care
                                        Physicians") where single tokens and pairs are
                                        too frequent, esp. for records with no address

Scoring: each shared key adds ``type_weight / block_size`` (block_size = number
of pool records carrying that key), so rare shared evidence dominates and the
many records that merely share a street or city add little. Keys whose block
exceeds a per-type cap are ignored. The best ``retrieve_k`` pairs per S1 per
source (S2, S3) are then scored by name + address token-set similarity
(rscore) and the final set per source is chosen by DEFAULT_CFG["rerank"]
(union of the best by blocking score and the best by rscore by default, so
records found only through rare keys, e.g. made-up names at the same address,
are kept alongside records that look alike). That final set is exactly what the
matching model scores (= candidate_pairs.tsv).
"""
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from src.config import N_JOBS
from src.normalize import ADDRESS_CANON, NAME_STOPWORDS

STREET_WORDS = sorted(set(ADDRESS_CANON.values()) | set(ADDRESS_CANON.keys()))
KEY_NAMES = {0: "N", 1: "P", 2: "A", 3: "U", 4: "W", 5: "C", 6: "B", 7: "M", 8: "T"}
TYPE_WEIGHT = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.0, 4: 1.0, 5: 2.0, 6: 2.0, 7: 2.0, 8: 2.5}
DEFAULT_CAPS = {0: 300, 1: 500, 2: 500, 3: 500, 4: 200, 5: 300, 6: 500, 7: 500, 8: 500}
# rerank: "none"  -> top final_k by blocking score
#         "combo" -> top final_k by rscore + min(bscore, 2) among the best retrieve_k
#         "union" -> best final_k//2 by bscore  UNION  best final_k//2 by rscore
#         "union3"-> best k_b by bscore UNION best k_r by rscore UNION best k_n by name only
# chosen with src.block_eval on real training data: recall 0.9647 at 24.8 candidates/S1
DEFAULT_CFG = {"caps": DEFAULT_CAPS, "weight": "inv", "retrieve_k": 50, "final_k": 16,
               "rerank": "union"}
_NO_T = {**DEFAULT_CAPS, 8: 0}
# Named blocking configurations (compared with src.block_eval; chosen with --blocking)
PRESETS = {
    "v6": {"caps": _NO_T, "weight": "inv", "retrieve_k": 50, "final_k": 16, "rerank": "union"},
    "v8": {"caps": DEFAULT_CAPS, "weight": "inv", "retrieve_k": 50, "final_k": 16, "rerank": "union"},
    "v9": {"caps": DEFAULT_CAPS, "weight": "inv", "retrieve_k": 50, "final_k": 0, "rerank": "union3",
           "k_b": 8, "k_r": 8, "k_n": 4},
    "v10": {"caps": DEFAULT_CAPS, "weight": "inv", "retrieve_k": 80, "final_k": 0, "rerank": "union3",
            "k_b": 8, "k_r": 8, "k_n": 4},
    "v11": {"caps": DEFAULT_CAPS, "weight": "inv", "retrieve_k": 80, "final_k": 0, "rerank": "union3",
            "k_b": 6, "k_r": 6, "k_n": 4},
    "v12": {"caps": DEFAULT_CAPS, "weight": "inv", "retrieve_k": 80, "final_k": 0, "rerank": "union3",
            "k_b": 8, "k_r": 8, "k_n": 8},
    "v13": {"caps": DEFAULT_CAPS, "weight": "inv", "retrieve_k": 80, "final_k": 0, "rerank": "union3",
            "k_b": 5, "k_r": 5, "k_n": 3},
}
MAX_NAME_TOKENS, MAX_ADDR_WORDS, MAX_NUMBERS = 4, 8, 3


def _token_tables(df: pl.DataFrame):
    """Flatten the token lists of records with an ``idx`` column (whole frame,
    never on a slice) -> (name tokens, address words, numbers, concat names)."""
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
    cc = b.select("idx", "name_concat").filter(pl.col("name_concat").str.len_chars() >= 4)
    return nt, aw, nu, cc


def _keys_from_tokens(nt, aw, nu, cc) -> pl.DataFrame:
    """Build the eight key types from flat token tables -> (idx, key:u64, t:u8)."""
    def keys(frame, parts, t):
        return frame.select("idx", pl.concat_str([pl.lit(KEY_NAMES[t] + "|")] + parts)
                            .alias("k"), pl.lit(t, pl.UInt8).alias("t"))

    sep = pl.lit("|")
    frames = [
        keys(nt, [pl.col("nt")], 0),
        keys(nt.join(nt, on="idx", suffix="2").filter(pl.col("nt") < pl.col("nt2")),
             [pl.col("nt"), sep, pl.col("nt2")], 1),
        keys(nt.join(aw, on="idx"), [pl.col("nt"), sep, pl.col("aw")], 2),
        keys(nu.join(aw, on="idx"), [pl.col("nu"), sep, pl.col("aw")], 3),
        keys(aw.join(aw, on="idx", suffix="2").filter(pl.col("aw") < pl.col("aw2")),
             [pl.col("aw"), sep, pl.col("aw2")], 4),
        keys(cc, [pl.col("name_concat")], 5),
        keys(nu.join(nt, on="idx"), [pl.col("nu"), sep, pl.col("nt")], 6),
        keys(nu.join(nu, on="idx", suffix="2").filter(pl.col("nu") < pl.col("nu2")),
             [pl.col("nu"), sep, pl.col("nu2")], 7),
        keys(nt.join(nt, on="idx", suffix="2").filter(pl.col("nt") < pl.col("nt2"))
             .join(nt.rename({"nt": "nt3"}), on="idx").filter(pl.col("nt2") < pl.col("nt3")),
             [pl.col("nt"), sep, pl.col("nt2"), sep, pl.col("nt3")], 8),
    ]
    return pl.concat(frames).select("idx", pl.col("k").hash(seed=11).alias("key"), "t")


def record_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Blocking keys for records with an ``idx`` column -> (idx, key:u64, t:u8)."""
    return _keys_from_tokens(*_token_tables(df))


def keys_chunked(df: pl.DataFrame, chunk: int = 1_000_000) -> pl.DataFrame:
    """record_keys for large frames, bounded memory.

    Token lists are flattened once on the whole frame; only the flat token
    tables are then processed in idx ranges. (Slicing list columns directly is
    unsafe: polars 1.35 mis-reads list columns of a sliced frame and silently
    built keys from the wrong rows.)
    """
    nt, aw, nu, cc = _token_tables(df)
    lo, hi = int(df["idx"].min()), int(df["idx"].max())
    parts = []
    for start in range(lo, hi + 1, chunk):
        rng = pl.col("idx").is_between(start, start + chunk - 1)
        parts.append(_keys_from_tokens(nt.filter(rng), aw.filter(rng), nu.filter(rng), cc.filter(rng)))
    return pl.concat(parts)


def pool_key_table(p: pl.DataFrame, max_cap: int) -> pl.DataFrame:
    """Pool keys with their block size ``df`` (keys with df > max_cap dropped)."""
    pk = keys_chunked(p)
    df = pk.group_by("key").agg(pl.len().cast(pl.UInt32).alias("df"))
    return (pk.join(df.filter(pl.col("df") <= max_cap), on="key")
            .select(pl.col("idx").alias("pidx"), "key", "t", "df"))


def _caps_frame(caps: dict) -> pl.DataFrame:
    return pl.DataFrame({"t": list(caps), "cap": list(caps.values()),
                         "tw": [TYPE_WEIGHT[t] for t in caps]},
                        schema={"t": pl.UInt8, "cap": pl.UInt32, "tw": pl.Float32})


def score_rows(rows: pl.DataFrame, caps: dict, weight: str) -> pl.DataFrame:
    """rows (idx, pidx, t, df) -> (idx, pidx, bscore, n_keytypes) under a config."""
    w = pl.col("tw") / (pl.col("df").cast(pl.Float32) if weight == "inv"
                        else (pl.col("df").cast(pl.Float32) + 1).log(2))
    return (rows.join(_caps_frame(caps), on="t").filter(pl.col("df") <= pl.col("cap"))
            .group_by("idx", "pidx")
            .agg(w.sum().cast(pl.Float32).alias("bscore"),
                 pl.col("t").n_unique().cast(pl.UInt8).alias("n_keytypes")))


def top_per_source(scored: pl.DataFrame, by: str, k: int) -> pl.DataFrame:
    """Keep the best ``k`` rows per (idx, source) ranked by column ``by``."""
    return (scored.sort(["idx", "source", by], descending=[False, False, True])
            .group_by(["idx", "source"], maintain_order=True).head(k))


def rerank_score(pairs: pl.DataFrame, q: pl.DataFrame, p: pl.DataFrame) -> pl.DataFrame:
    """Add rscore = name token-set + address token-set similarity (0..2)."""
    d = (pairs.join(q.select("idx", pl.col("name_core").alias("n1"), pl.col("addr_clean").alias("a1")),
                    on="idx")
         .join(p.select(pl.col("idx").alias("pidx"), pl.col("name_core").alias("n2"),
                        pl.col("addr_clean").alias("a2")), on="pidx"))
    ns = cpdist(d["n1"].to_list(), d["n2"].to_list(), scorer=fuzz.token_set_ratio,
                workers=N_JOBS, dtype=np.float32)
    ad = cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio,
                workers=N_JOBS, dtype=np.float32)
    return d.drop("n1", "n2", "a1", "a2").with_columns(
        pl.Series("rscore", (ns + ad) / np.float32(100.0)),
        pl.Series("nscore", ns / np.float32(100.0)))


def select_final(scored: pl.DataFrame, q: pl.DataFrame, p: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Apply the configured final selection to scored pairs (idx, pidx, source, bscore, ...)."""
    mode, k = cfg["rerank"], cfg["final_k"]
    if mode in (None, False, "none"):
        return top_per_source(scored, "bscore", k)
    pre = rerank_score(top_per_source(scored, "bscore", cfg["retrieve_k"]), q, p)
    if mode == "combo":
        pre = pre.with_columns((pl.col("rscore") + pl.col("bscore").clip(upper_bound=2.0)).alias("combo"))
        return top_per_source(pre, "combo", k).drop("combo")
    if mode == "union":
        half = max(k // 2, 1)
        a = top_per_source(pre, "bscore", half)
        b = top_per_source(pre, "rscore", half)
        return pl.concat([a, b]).unique(["idx", "pidx"], keep="first")
    if mode == "union3":
        # best k_b by blocking score, best k_r by name+address similarity and best k_n
        # by name similarity alone (records with an empty/garbled address are
        # otherwise always outranked by other businesses at the S1's address)
        parts = [top_per_source(pre, "bscore", cfg["k_b"]), top_per_source(pre, "rscore", cfg["k_r"]),
                 top_per_source(pre, "nscore", cfg["k_n"])]
        return pl.concat(parts).unique(["idx", "pidx"], keep="first")
    return top_per_source(pre, "rscore", k)                      # "rscore"


def generate_candidates(queries: pl.DataFrame, pool: pl.DataFrame, cfg: dict = None,
                        chunk_size: int = 20_000, log=print, top_k: int = None) -> pl.DataFrame:
    """Candidate pairs for every query (S1) record against the S2/S3 pool.

    Returns columns: source1_entity_id, cand_id, source, bscore, n_keytypes.
    Pairs are only formed within the same ``country`` label (EDA: 0 of 7.6M
    training matches cross countries).
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    if top_k is not None:
        cfg["final_k"] = top_k
    caps = {int(k): v for k, v in cfg["caps"].items()}
    out = []
    for country in sorted(queries["country"].unique().to_list()):
        t0 = time.time()
        q = queries.filter(pl.col("country") == country).with_row_index("idx")
        p = pool.filter(pl.col("country") == country).with_row_index("idx")
        if p.height == 0:
            log(f"  [{country}] no pool records")
            continue
        pk = pool_key_table(p, max(caps.values()))
        qk = keys_chunked(q).drop("t")
        src = p.select(pl.col("idx").alias("pidx"), pl.col("entity_id").str.slice(0, 2).alias("source"))
        n_pairs = 0
        for start in range(0, q.height, chunk_size):
            qc = qk.filter(pl.col("idx").is_between(start, start + chunk_size - 1))
            scored = score_rows(qc.join(pk, on="key"), caps, cfg["weight"]).join(src, on="pidx")
            cand = (select_final(scored, q, p, cfg).join(q.select("idx", pl.col("entity_id").alias("source1_entity_id")), on="idx")
                    .join(p.select(pl.col("idx").alias("pidx"), pl.col("entity_id").alias("cand_id")),
                          on="pidx")
                    .select("source1_entity_id", "cand_id", "source", "bscore", "n_keytypes"))
            n_pairs += cand.height
            out.append(cand)
        log(f"  [{country}] {q.height:,} queries x {p.height:,} pool -> {n_pairs:,} pairs "
            f"({n_pairs / max(q.height, 1):.1f}/query) in {time.time() - t0:.0f}s")
    if not out:
        return pl.DataFrame(schema={"source1_entity_id": pl.Utf8, "cand_id": pl.Utf8,
                                    "source": pl.Utf8, "bscore": pl.Float32,
                                    "n_keytypes": pl.UInt8})
    return pl.concat(out)


def blocking_recall(cands: pl.DataFrame, gt_pairs: pl.DataFrame, query_ids) -> dict:
    """Share of true (S1, S2/S3) pairs of the given queries present in ``cands``."""
    ids = pl.Series(query_ids).implode()
    truth = gt_pairs.filter(pl.col("source1_entity_id").is_in(ids))
    found = truth.join(cands.select("source1_entity_id", pl.col("cand_id").alias("matched_id")),
                       on=["source1_entity_id", "matched_id"], how="semi").height
    n_q = pl.Series(query_ids).n_unique()
    return {"true_pairs": truth.height, "found": found,
            "recall": round(found / max(truth.height, 1), 4),
            "cands_per_query": round(cands.height / max(n_q, 1), 2)}
