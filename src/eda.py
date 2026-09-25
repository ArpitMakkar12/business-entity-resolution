"""Exploratory data analysis for the Business Entity Resolution challenge.

Answers the questions that drive the pipeline design:
  * how many matches each S1 entity has, and how many are singletons
  * whether each S2/S3 record belongs to at most one S1 entity (exclusivity)
  * how many S2/S3 records match nothing (distractor rate), train vs test
  * whether matched records always share the S1 record's country label
  * missing values, scripts (Devanagari/Tamil/...), postal-code patterns

Usage (from the business_entity_resolution/ folder):
    python -m src.eda --out-dir work/eda                    # full data
    python -m src.eda --out-dir work/eda_sample --max-rows 300000   # laptop

--max-rows caps the rows read from each SOURCE file (the ground truth is
always read in full). Statistics that join ground truth to source records are
then computed on the loaded subset only and are marked "partial" in the report.

Outputs in --out-dir: eda_report.md (read this), eda_report.json,
match_examples.tsv (matched groups side by side), singleton_examples.tsv.
Reads only the provided files; no network access.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import polars as pl

from src.config import DATA_DIR, SEED, test_paths, train_paths
from src.io_utils import explode_ground_truth, read_ground_truth, read_source

# Polars 1.x warns about a future explode() default; our lists are never empty here.
warnings.filterwarnings("ignore", message=".*empty_as_null.*")

# Unicode script checks (Rust regex syntax, used by polars).
INDIC = r"[\p{Devanagari}\p{Tamil}\p{Telugu}\p{Kannada}\p{Malayalam}\p{Bengali}\p{Gujarati}\p{Gurmukhi}\p{Oriya}]"
ACCENTED_LATIN = r"[\x{00C0}-\x{024F}]"
NON_ASCII = r"[^\x00-\x7F]"


def _pct(x: float) -> float:
    return round(100.0 * float(x), 2)


def source_stats(df: pl.DataFrame) -> dict:
    """Per-file statistics: size, duplicates, countries, missingness, scripts."""
    name = pl.col("business_name").fill_null("")
    addr = pl.col("business_address").fill_null("")
    per_country = (
        df.group_by("country")
        .agg(
            pl.len().alias("rows"),
            (name.str.strip_chars() == "").mean().alias("name_empty"),
            (addr.str.strip_chars() == "").mean().alias("addr_empty"),
            addr.str.to_lowercase().str.contains(r"\bnull\b").mean().alias("addr_has_null_literal"),
            name.str.contains(INDIC).mean().alias("name_indic_script"),
            addr.str.contains(INDIC).mean().alias("addr_indic_script"),
            name.str.contains(ACCENTED_LATIN).mean().alias("name_accented_latin"),
            name.str.contains(NON_ASCII).mean().alias("name_any_non_ascii"),
            addr.str.contains(r"\d").mean().alias("addr_has_digit"),
            addr.str.contains(r"(^|\D)\d{5}(\D|$)").mean().alias("addr_5digit_token"),
            addr.str.contains(r"(^|\D)\d{6}(\D|$)").mean().alias("addr_6digit_token"),
            name.str.contains(r"(?i)(\.com|\.in|\.fr|www\.)").mean().alias("name_has_domain"),
            name.str.len_chars().mean().alias("name_len_mean"),
            addr.str.len_chars().mean().alias("addr_len_mean"),
        )
        .sort("rows", descending=True)
    )
    out = {
        "rows": df.height,
        "unique_ids": df["entity_id"].n_unique(),
        "duplicate_id_rows": df.height - df["entity_id"].n_unique(),
        "country_null": int(df["country"].null_count()),
        "per_country": {},
    }
    for row in per_country.iter_rows(named=True):
        c = row.pop("country") or "<null>"
        out["per_country"][c] = {
            k: (v if k in ("rows",) else round(v, 2) if k.endswith("_mean") else _pct(v))
            for k, v in row.items()
        }
    return out


def ground_truth_stats(gt: pl.DataFrame, pairs: pl.DataFrame) -> dict:
    """Match-count distribution, singleton rate and the exclusivity check."""
    n = gt.height
    dist = (
        gt.with_columns(pl.col("n_matches").clip(upper_bound=10).alias("k"))
        .group_by("k").len().sort("k")
    )
    per_src = gt.with_columns(
        pl.col("matched").list.eval(pl.element().str.starts_with("S2-")).list.sum().alias("n_s2"),
        pl.col("matched").list.eval(pl.element().str.starts_with("S3-")).list.sum().alias("n_s3"),
    )
    id_counts = pairs.group_by("matched_id").len()
    multi = id_counts.filter(pl.col("len") > 1)
    bad_prefix = pairs.filter(~pl.col("matched_id").str.contains(r"^S[23]-")).height
    return {
        "s1_rows": n,
        "duplicate_s1_rows": n - gt["source1_entity_id"].n_unique(),
        "singleton_rate_pct": _pct((gt["n_matches"] == 0).mean()),
        "matches_mean": round(gt["n_matches"].mean(), 3),
        "matches_median": float(gt["n_matches"].median()),
        "matches_max": int(gt["n_matches"].max()),
        "n_matches_distribution (10 = 10+)": {int(k): int(v) for k, v in dist.iter_rows()},
        "s2_per_s1_mean": round(per_src["n_s2"].mean(), 3),
        "s3_per_s1_mean": round(per_src["n_s3"].mean(), 3),
        "pct_s1_with_no_s2": _pct((per_src["n_s2"] == 0).mean()),
        "pct_s1_with_no_s3": _pct((per_src["n_s3"] == 0).mean()),
        "positive_pairs": pairs.height,
        "distinct_matched_ids": id_counts.height,
        "EXCLUSIVITY_ids_in_more_than_one_s1": multi.height,
        "EXCLUSIVITY_holds": multi.height == 0,
        "matched_ids_with_bad_prefix": bad_prefix,
    }


def join_stats(gt, pairs, s1, s2, s3, partial: bool) -> dict:
    """Stats that need source records: coverage, country agreement, per-country."""
    s23 = pl.concat([s2, s3])
    matched_set = pairs.select(pl.col("matched_id").alias("entity_id")).unique()
    cov = (
        s23.join(matched_set.with_columns(pl.lit(True).alias("is_matched")),
                 on="entity_id", how="left")
        .with_columns(pl.col("is_matched").fill_null(False),
                      pl.col("entity_id").str.slice(0, 2).alias("src"))
        .group_by("src", "country")
        .agg(pl.len().alias("rows"), pl.col("is_matched").mean().alias("matched_share"))
        .sort("src", "country")
    )
    coverage = {f"{r['src']}|{r['country']}": {"rows": r["rows"],
                "pct_matched_to_some_s1": _pct(r["matched_share"])}
                for r in cov.iter_rows(named=True)}

    p = (
        pairs.join(s1.select(pl.col("entity_id").alias("source1_entity_id"),
                             pl.col("country").alias("c1")), on="source1_entity_id", how="inner")
        .join(s23.select(pl.col("entity_id").alias("matched_id"),
                         pl.col("country").alias("c2")), on="matched_id", how="inner")
    )
    mismatch = p.filter(pl.col("c1") != pl.col("c2"))

    per_c = (
        gt.join(s1.select(pl.col("entity_id").alias("source1_entity_id"), "country"),
                on="source1_entity_id", how="inner")
        .group_by("country")
        .agg(pl.len().alias("s1_rows"),
             (pl.col("n_matches") == 0).mean().alias("singleton"),
             pl.col("n_matches").mean().alias("matches_mean"))
        .sort("s1_rows", descending=True)
    )
    return {
        "partial_because_max_rows": partial,
        "s2_s3_coverage_by_country": coverage,
        "pairs_checked_for_country": p.height,
        "pairs_with_country_mismatch": mismatch.height,
        "country_mismatch_examples": mismatch.head(10).rows(),
        "gt_by_s1_country": {
            r["country"]: {"s1_rows": r["s1_rows"], "singleton_pct": _pct(r["singleton"]),
                           "matches_mean": round(r["matches_mean"], 3)}
            for r in per_c.iter_rows(named=True)},
    }


def examples(gt, s1, s2, s3, n_groups: int = 40, n_single: int = 25):
    """Side-by-side matched groups and singleton examples for eyeballing noise."""
    s23 = pl.concat([s2, s3])
    loaded_s1 = gt.join(s1.select(pl.col("entity_id").alias("source1_entity_id")),
                        on="source1_entity_id", how="inner")
    with_matches = loaded_s1.filter(pl.col("n_matches") > 0)
    groups = with_matches.sample(min(n_groups, with_matches.height), seed=SEED)
    long = (
        groups.select("source1_entity_id", "matched").explode("matched")
        .rename({"matched": "entity_id"})
    )
    rows = pl.concat([
        groups.select(pl.col("source1_entity_id"), pl.col("source1_entity_id").alias("entity_id")),
        long,
    ]).join(pl.concat([s1, s23]), on="entity_id", how="left")
    rows = rows.sort("source1_entity_id", "entity_id")

    singles = loaded_s1.filter(pl.col("n_matches") == 0)
    singles = singles.sample(min(n_single, singles.height), seed=SEED).join(
        s1, left_on="source1_entity_id", right_on="entity_id", how="left")
    return rows, singles.drop("matched")


def to_markdown(report: dict) -> str:
    """Render the JSON report as readable markdown."""
    lines = ["# EDA report", "", f"Generated in {report['runtime_sec']} s. "
             f"max_rows={report['max_rows']}", ""]

    def table(d: dict, title: str):
        lines.extend([f"## {title}", ""])
        for k, v in d.items():
            if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
                cols = list(next(iter(v.values())).keys())
                lines.append(f"**{k}**\n")
                lines.append("| key | " + " | ".join(cols) + " |")
                lines.append("|---|" + "---|" * len(cols))
                for kk, vv in v.items():
                    lines.append(f"| {kk} | " + " | ".join(str(vv.get(c)) for c in cols) + " |")
                lines.append("")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    for section in ("ground_truth", "train_join_stats"):
        if section in report:
            table(report[section], section)
    for split in ("train_sources", "test_sources"):
        for name, st in report.get(split, {}).items():
            table(st, f"{split} / {name}")
    if "test_ratios" in report:
        table(report["test_ratios"], "train vs test density")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out-dir", default="work/eda")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap rows read per source file (laptop mode)")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr, te = train_paths(args.data_dir), test_paths(args.data_dir)
    report = {"max_rows": args.max_rows}

    print("[1/5] reading train sources ...", flush=True)
    s1, s2, s3 = (read_source(tr[k], args.max_rows) for k in ("s1", "s2", "s3"))
    print("[2/5] reading ground truth ...", flush=True)
    gt = read_ground_truth(tr["gt"])
    pairs = explode_ground_truth(gt)

    print("[3/5] ground-truth + source statistics ...", flush=True)
    report["ground_truth"] = ground_truth_stats(gt, pairs)
    report["train_sources"] = {k: source_stats(d) for k, d in (("s1", s1), ("s2", s2), ("s3", s3))}
    report["train_join_stats"] = join_stats(gt, pairs, s1, s2, s3, partial=args.max_rows is not None)

    rows, singles = examples(gt, s1, s2, s3)
    rows.write_csv(out / "match_examples.tsv", separator="\t")
    singles.write_csv(out / "singleton_examples.tsv", separator="\t")

    if not args.skip_test:
        print("[4/5] reading test sources ...", flush=True)
        t1, t2, t3 = (read_source(te[k], args.max_rows) for k in ("s1", "s2", "s3"))
        report["test_sources"] = {k: source_stats(d) for k, d in (("s1", t1), ("s2", t2), ("s3", t3))}
        ratio = {}
        for split, (a, b, c) in (("train", (s1, s2, s3)), ("test", (t1, t2, t3))):
            for country in sorted(set(a["country"].drop_nulls().unique().to_list())):
                n1 = a.filter(pl.col("country") == country).height
                n23 = (b.filter(pl.col("country") == country).height
                       + c.filter(pl.col("country") == country).height)
                ratio[f"{split}|{country}"] = {"s1": n1, "s2+s3": n23,
                                               "s2s3_per_s1": round(n23 / max(n1, 1), 3)}
        report["test_ratios"] = {"by_split_country": ratio}

    print("[5/5] writing report ...", flush=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    (out / "eda_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    (out / "eda_report.md").write_text(to_markdown(report), encoding="utf-8")
    g = report["ground_truth"]
    print(f"\nDone in {report['runtime_sec']} s -> {out.resolve()}")
    print(f"singleton rate: {g['singleton_rate_pct']}% | mean matches: {g['matches_mean']} | "
          f"exclusivity holds: {g['EXCLUSIVITY_holds']}")


if __name__ == "__main__":
    main()
