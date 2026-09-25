"""Reading the challenge TSVs and writing submission files.

All readers use an explicit tab separator and disable quote handling, because
names/addresses may contain quote characters that must be kept verbatim.
Every column is read as a string (entity IDs must never become numbers).
"""
from pathlib import Path
from typing import Optional

import polars as pl

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def _read_tsv(path: Path, n_rows: Optional[int] = None) -> pl.DataFrame:
    """Read a TSV as all-string columns; empty fields become null."""
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema=False,
        has_header=True,
        n_rows=n_rows,
        truncate_ragged_lines=True,
    )


def read_source(path: Path, n_rows: Optional[int] = None) -> pl.DataFrame:
    """Read one source file (S1/S2/S3) and validate its columns."""
    df = _read_tsv(path, n_rows)
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {df.columns}")
    return df.select(SOURCE_COLUMNS)


def read_ground_truth(path: Path) -> pl.DataFrame:
    """Read the ground truth and parse the comma-separated ID list.

    Returns columns: source1_entity_id, matched (List[str]), n_matches.
    An empty matched_entity_ids field becomes an empty list (a singleton).
    """
    df = _read_tsv(path)
    missing = set(GT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {df.columns}")
    return df.select(
        pl.col("source1_entity_id"),
        pl.col("matched_entity_ids")
        .fill_null("")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars())
        .list.eval(pl.element().filter(pl.element() != ""))
        .alias("matched"),
    ).with_columns(pl.col("matched").list.len().alias("n_matches"))


def explode_ground_truth(gt: pl.DataFrame) -> pl.DataFrame:
    """One row per (S1, matched ID) positive pair; singletons are dropped."""
    return (
        gt.filter(pl.col("n_matches") > 0)
        .select("source1_entity_id", "matched")
        .explode("matched")
        .drop_nulls("matched")
        .rename({"matched": "matched_id"})
    )


def write_id_lists(mapping: pl.DataFrame, path: Path, list_col_name: str) -> None:
    """Write a submission-style TSV (matching_results / candidate_pairs).

    ``mapping`` needs columns ``source1_entity_id`` and ``ids`` (List[str]).
    Written by hand (not via a CSV writer) so there is no quoting, no
    trailing whitespace, and empty lists produce an empty second field.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{list_col_name}\n")
        for s1, ids in mapping.select("source1_entity_id", "ids").iter_rows():
            uniq = list(dict.fromkeys(ids or []))  # de-duplicate, keep order
            f.write(f"{s1.strip()}\t{','.join(uniq)}\n")
