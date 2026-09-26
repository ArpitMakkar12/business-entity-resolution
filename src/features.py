"""Pair features for the matching model.

For every candidate pair (S1 record, S2/S3 record) we compute country-agnostic
similarity features (no country one-hot, so the model transfers to France):

Name      Jaro-Winkler, token-set/sort/partial ratios on the core name,
          ratios on the concatenated name (handles/domains) and phonetic
          skeleton, DBA-alias similarity, token overlap/containment, IDF-
          weighted overlap (rarity of shared and missing tokens), legal-form
          agreement, record flags (handle, domain, DBA, Indic script).
Address   token-set/ratio similarity, token Jaccard/containment, house-number
          overlap (incl. first number match), state overlap, emptiness flags.
Context   blocking score and key types, rank of the pair inside its S1 group,
          gaps to the best candidate of the group, number of candidates, and
          duplicate support (other candidates sharing house number / name).

String similarities use rapidfuzz.process.cpdist (C++, multi-threaded).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from src.config import N_JOBS

RECORD_COLS = ["entity_id", "name_core", "name_concat", "name_skeleton", "name_alt_core",
               "name_legal", "name_core_tokens", "name_is_handle", "name_is_domain",
               "name_has_dba", "name_has_indic", "addr_clean", "addr_tokens",
               "addr_numbers", "addr_states", "addr_empty", "country", "name_freq"]

FEATURES = [
    # context
    "bscore", "n_keytypes", "src_is_s3", "rank_in_src", "n_cands", "bscore_gap",
    # name
    "name_jw", "name_tset", "name_tsort", "name_partial", "concat_ratio", "concat_partial",
    "skel_ratio", "alt_tset", "ntok_inter", "ntok_jacc", "ntok_cont1", "ntok_cont2",
    "idf_shared_sum", "idf_shared_max", "idf_frac1", "idf_frac2", "idf_miss1_max",
    "idf_miss2_max", "legal_eq", "legal2_empty", "handle2", "domain2", "dba2", "indic2",
    "len_core1", "len_core2", "name_tset_gap",
    # address
    "addr_tset", "addr_ratio", "atok_inter", "atok_jacc", "atok_cont1", "atok_cont2",
    "num_inter", "num_jacc", "num_first_match", "n_nums1", "n_nums2", "addr_empty2",
    "state_overlap", "state_both", "addr_tset_gap",
    # duplicate support inside the S1 group
    "grp_same_num", "grp_same_concat",
    # sibling-business signals (neighbouring house number, one name word changed)
    "num_conflict", "num_delta_log", "num_delta_small", "ntok_only1", "ntok_only2",
    "name_one_sub", "grp_s1num_support", "grp_num_minority",
    # name ambiguity: how many records in the same table/country share the exact name
    "s1_name_freq", "cand_name_freq", "addr_empty_amb",
]


def token_idf(*frames: pl.DataFrame) -> pl.DataFrame:
    """IDF of name core tokens per country over the given record frames."""
    allr = pl.concat([f.select("country", "name_core_tokens") for f in frames])
    n_docs = allr.group_by("country").len().rename({"len": "n_docs"})
    df = (allr.explode("name_core_tokens").drop_nulls("name_core_tokens")
          .group_by("country", "name_core_tokens").len())  # tokens are unique per record
    return (df.join(n_docs, on="country")
            .select("country", pl.col("name_core_tokens").alias("tok"),
                    (pl.col("n_docs") / pl.col("len")).log().cast(pl.Float32).alias("idf")))


def add_name_freq(df: pl.DataFrame) -> pl.DataFrame:
    """name_freq = number of records in ``df`` (same country) with the same concatenated name.

    Computed on a *whole* table (all S1 records, or the whole S2+S3 pool) so the
    value is comparable between training and test.
    """
    return df.with_columns(pl.len().over(["country", "name_concat"]).cast(pl.UInt32).alias("name_freq"))


def _min_abs_delta(s: pl.Series) -> pl.Series:
    """Per row: min |n1 - n| over the candidate's numeric house numbers (null if none)."""
    n1 = s.struct.field("n1")
    ns = s.struct.field("ns")
    ex = pl.DataFrame({"i": range(len(s)), "n1": n1, "ns": ns}).explode("ns")
    agg = (ex.with_columns((pl.col("ns") - pl.col("n1")).abs().alias("d"))
           .group_by("i").agg(pl.col("d").min()).sort("i"))
    return agg["d"].cast(pl.Float64)


def attach_records(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """Join S1 (suffix _1) and candidate (suffix _2) record views onto the pairs."""
    a = s1.select(RECORD_COLS).rename({c: f"{c}_1" for c in RECORD_COLS})
    b = pool.select(RECORD_COLS).rename({c: f"{c}_2" for c in RECORD_COLS})
    return (pairs.join(a, left_on="source1_entity_id", right_on="entity_id_1")
            .join(b.drop("country_2"), left_on="cand_id", right_on="entity_id_2"))


def _cp(df: pl.DataFrame, a: str, b: str, scorer, scale: float = 100.0) -> np.ndarray:
    return cpdist(df[a].fill_null("").to_list(), df[b].fill_null("").to_list(),
                  scorer=scorer, workers=N_JOBS, dtype=np.float32) / np.float32(scale)


def _idf_features(df: pl.DataFrame, idf: pl.DataFrame) -> pl.DataFrame:
    """IDF-weighted token overlap features, keyed by the pair id ``pid``."""
    max_idf = idf.group_by("country").agg(pl.col("idf").max().alias("max_idf"))

    def expl(col):
        return (df.select("pid", pl.col("country_1").alias("country"), pl.col(col).alias("tok"))
                .explode("tok").drop_nulls("tok")
                .join(idf, on=["country", "tok"], how="left").join(max_idf, on="country")
                .with_columns(pl.col("idf").fill_null(pl.col("max_idf"))).drop("max_idf"))

    e1, e2 = expl("name_core_tokens_1"), expl("name_core_tokens_2")
    keys = ["pid", "tok"]
    shared = e1.join(e2.select(keys), on=keys, how="semi")
    parts = [
        e1.group_by("pid").agg(pl.col("idf").sum().alias("idf1_sum")),
        e2.group_by("pid").agg(pl.col("idf").sum().alias("idf2_sum")),
        shared.group_by("pid").agg(pl.col("idf").sum().alias("idf_shared_sum"),
                                   pl.col("idf").max().alias("idf_shared_max")),
        e1.join(e2.select(keys), on=keys, how="anti").group_by("pid")
        .agg(pl.col("idf").max().alias("idf_miss1_max")),
        e2.join(e1.select(keys), on=keys, how="anti").group_by("pid")
        .agg(pl.col("idf").max().alias("idf_miss2_max")),
    ]
    out = df.select("pid")
    for p in parts:
        out = out.join(p, on="pid", how="left")
    return out.with_columns(pl.exclude("pid").fill_null(0.0)).with_columns(
        (pl.col("idf_shared_sum") / pl.col("idf1_sum").clip(lower_bound=1e-6)).alias("idf_frac1"),
        (pl.col("idf_shared_sum") / pl.col("idf2_sum").clip(lower_bound=1e-6)).alias("idf_frac2"),
    ).drop("idf1_sum", "idf2_sum")


def compute_features(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame,
                     idf: pl.DataFrame) -> pl.DataFrame:
    """Return pairs (source1_entity_id, cand_id, source, ...) + FEATURES columns."""
    df = attach_records(pairs, s1, pool).with_row_index("pid")
    sim = {
        "name_jw": _cp(df, "name_core_1", "name_core_2", JaroWinkler.normalized_similarity, 1.0),
        "name_tset": _cp(df, "name_core_1", "name_core_2", fuzz.token_set_ratio),
        "name_tsort": _cp(df, "name_core_1", "name_core_2", fuzz.token_sort_ratio),
        "name_partial": _cp(df, "name_core_1", "name_core_2", fuzz.partial_ratio),
        "concat_ratio": _cp(df, "name_concat_1", "name_concat_2", fuzz.ratio),
        "concat_partial": _cp(df, "name_concat_1", "name_concat_2", fuzz.partial_ratio),
        "skel_ratio": _cp(df, "name_skeleton_1", "name_skeleton_2", fuzz.ratio),
        "alt_tset": _cp(df, "name_core_1", "name_alt_core_2", fuzz.token_set_ratio),
        "addr_tset": _cp(df, "addr_clean_1", "addr_clean_2", fuzz.token_set_ratio),
        "addr_ratio": _cp(df, "addr_clean_1", "addr_clean_2", fuzz.ratio),
    }
    df = df.with_columns([pl.Series(k, v) for k, v in sim.items()])

    def inter(a, b):
        return pl.col(a).list.set_intersection(pl.col(b)).list.len().cast(pl.Float32)

    def union(a, b):
        return pl.col(a).list.set_union(pl.col(b)).list.len().cast(pl.Float32).clip(lower_bound=1)

    L = lambda c: pl.col(c).list.len().cast(pl.Float32)  # noqa: E731
    df = df.with_columns(
        (pl.col("source") == "S3").cast(pl.Int8).alias("src_is_s3"),
        inter("name_core_tokens_1", "name_core_tokens_2").alias("ntok_inter"),
        (inter("name_core_tokens_1", "name_core_tokens_2")
         / union("name_core_tokens_1", "name_core_tokens_2")).alias("ntok_jacc"),
        (inter("name_core_tokens_1", "name_core_tokens_2")
         / L("name_core_tokens_1").clip(lower_bound=1)).alias("ntok_cont1"),
        (inter("name_core_tokens_1", "name_core_tokens_2")
         / L("name_core_tokens_2").clip(lower_bound=1)).alias("ntok_cont2"),
        (pl.col("name_legal_1") == pl.col("name_legal_2")).cast(pl.Int8).alias("legal_eq"),
        (pl.col("name_legal_2") == "").cast(pl.Int8).alias("legal2_empty"),
        pl.col("name_is_handle_2").cast(pl.Int8).alias("handle2"),
        pl.col("name_is_domain_2").cast(pl.Int8).alias("domain2"),
        pl.col("name_has_dba_2").cast(pl.Int8).alias("dba2"),
        pl.col("name_has_indic_2").cast(pl.Int8).alias("indic2"),
        (pl.col("name_freq_1").cast(pl.Float32) + 1).log(10).alias("s1_name_freq"),
        (pl.col("name_freq_2").cast(pl.Float32) + 1).log(10).alias("cand_name_freq"),
        # an address-less record whose name is shared by several S1 entities is ambiguous
        (pl.col("addr_empty_2") & (pl.col("name_freq_1") > 1)).cast(pl.Int8).alias("addr_empty_amb"),
        pl.col("name_core_1").str.len_chars().cast(pl.Float32).alias("len_core1"),
        pl.col("name_core_2").str.len_chars().cast(pl.Float32).alias("len_core2"),
        inter("addr_tokens_1", "addr_tokens_2").alias("atok_inter"),
        (inter("addr_tokens_1", "addr_tokens_2") / union("addr_tokens_1", "addr_tokens_2"))
        .alias("atok_jacc"),
        (inter("addr_tokens_1", "addr_tokens_2") / L("addr_tokens_1").clip(lower_bound=1))
        .alias("atok_cont1"),
        (inter("addr_tokens_1", "addr_tokens_2") / L("addr_tokens_2").clip(lower_bound=1))
        .alias("atok_cont2"),
        inter("addr_numbers_1", "addr_numbers_2").alias("num_inter"),
        (inter("addr_numbers_1", "addr_numbers_2") / union("addr_numbers_1", "addr_numbers_2"))
        .alias("num_jacc"),
        pl.col("addr_numbers_2").list.contains(pl.col("addr_numbers_1").list.first())
        .fill_null(False).cast(pl.Int8).alias("num_first_match"),
        L("addr_numbers_1").alias("n_nums1"),
        L("addr_numbers_2").alias("n_nums2"),
        pl.col("addr_empty_2").cast(pl.Int8).alias("addr_empty2"),
        (inter("addr_states_1", "addr_states_2") > 0).cast(pl.Int8).alias("state_overlap"),
        ((L("addr_states_1") > 0) & (L("addr_states_2") > 0)).cast(pl.Int8).alias("state_both"),
        pl.col("addr_numbers_2").list.first().fill_null("").alias("_num2"),
        # both sides carry house numbers but share none
        ((L("addr_numbers_1") > 0) & (L("addr_numbers_2") > 0)
         & (inter("addr_numbers_1", "addr_numbers_2") == 0)).cast(pl.Int8).alias("num_conflict"),
        # name tokens present on only one side
        (L("name_core_tokens_1") - inter("name_core_tokens_1", "name_core_tokens_2")).alias("ntok_only1"),
        (L("name_core_tokens_2") - inter("name_core_tokens_1", "name_core_tokens_2")).alias("ntok_only2"),
        pl.col("addr_numbers_1").list.first().alias("_num1"),
    )
    # distance between S1's first house number and the closest candidate number
    d = pl.col("addr_numbers_2").list.eval(
        pl.element().cast(pl.Int64, strict=False)).list.eval(pl.element().drop_nulls())
    df = df.with_columns(
        pl.struct(n1=pl.col("_num1").cast(pl.Int64, strict=False), ns=d)
        .map_batches(_min_abs_delta, return_dtype=pl.Float64).alias("_delta"))
    df = df.with_columns(
        pl.when(pl.col("_delta").is_null()).then(-1.0)
        .otherwise((pl.col("_delta") + 1).log(10)).alias("num_delta_log"),
        (pl.col("_delta").is_between(1, 30).fill_null(False)).cast(pl.Int8).alias("num_delta_small"),
        ((pl.col("ntok_only1") == 1) & (pl.col("ntok_only2") == 1)).cast(pl.Int8).alias("name_one_sub"),
        pl.col("addr_numbers_2").list.contains(pl.col("_num1")).fill_null(False).alias("_has_s1num"),
    )
    g = "source1_entity_id"
    df = df.with_columns(
        pl.col("bscore").rank("ordinal", descending=True).over([g, "source"])
        .cast(pl.Float32).alias("rank_in_src"),
        pl.len().over(g).cast(pl.Float32).alias("n_cands"),
        (pl.col("bscore").max().over(g) - pl.col("bscore")).alias("bscore_gap"),
        (pl.col("name_tset").max().over(g) - pl.col("name_tset")).alias("name_tset_gap"),
        (pl.col("addr_tset").max().over(g) - pl.col("addr_tset")).alias("addr_tset_gap"),
        (pl.when(pl.col("_num2") != "").then(pl.len().over([g, "_num2"]) - 1).otherwise(0))
        .cast(pl.Float32).alias("grp_same_num"),
        (pl.len().over([g, "name_concat_2"]) - 1).cast(pl.Float32).alias("grp_same_concat"),
        # how many other candidates carry the S1 record's own house number
        (pl.col("_has_s1num").cast(pl.Int32).sum().over(g) - pl.col("_has_s1num").cast(pl.Int32))
        .cast(pl.Float32).alias("grp_s1num_support"),
    )
    df = df.with_columns(
        # candidate lacks the S1 number while other candidates in the group have it
        ((~pl.col("_has_s1num")) & (pl.col("_num1").is_not_null())
         & (pl.col("grp_s1num_support") > 0)).cast(pl.Int8).alias("grp_num_minority"))
    df = df.join(_idf_features(df, idf), on="pid", how="left")
    return df.select(["source1_entity_id", "cand_id", "source", "country_1"]
                     + [pl.col(f).cast(pl.Float32) for f in FEATURES]).rename({"country_1": "country"})


def compute_features_chunked(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame,
                             idf: pl.DataFrame, s1_per_chunk: int = 100_000, log=print):
    """Yield feature frames for consecutive chunks of S1 records (bounded memory)."""
    ids = pairs.select("source1_entity_id").unique(maintain_order=True)["source1_entity_id"]
    for i in range(0, len(ids), s1_per_chunk):
        chunk_ids = ids.slice(i, s1_per_chunk)
        part = pairs.filter(pl.col("source1_entity_id").is_in(chunk_ids.implode()))
        s1c = s1.filter(pl.col("entity_id").is_in(chunk_ids.implode()))
        poolc = pool.filter(pl.col("entity_id").is_in(part["cand_id"].unique().implode()))
        yield compute_features(part, s1c, poolc, idf)
        log(f"  features: {min(i + s1_per_chunk, len(ids)):,}/{len(ids):,} S1 records")
