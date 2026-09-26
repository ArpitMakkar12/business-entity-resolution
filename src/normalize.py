"""Normalization of business names and addresses into comparable views.

Source 1 is clean (Latin script, no placeholders); all noise lives in Source 2/3.
Every record, from any source and any country, is mapped onto the same
canonical views so that blocking keys and similarity features compare like
with like. The rules come from noise patterns observed in the training data:

Names
  * legal forms moved, bracketed, abbreviated, added or typo'd
    ("PRVHTE RAJ HEALTHCARE [LIMITED]", "Inc. Pediatric Dental", "L.L.C.")
  * DBA / F/K/A / "trading as" with an invented prefix name; "name | www.x.com"
  * domains and social handles ("navaportillo.com", "@alikadavis")
  * junk prefixes/suffixes (">>", "...", "M/s", "#66553"), injected accents,
    OCR digit swaps ("C1ark", "5atpura"), repeated tokens, hyphen joins
  * Indic-script versions (Devanagari, Kannada, Telugu, Tamil, ...)
Addresses
  * reordered components, case, street-type abbreviations, city aliases
  * "<NULL>", "null", "N/A" fillers; "##", "#", "H.no", "Plot No" designators
  * house numbers with leading zeros, trailing '.'/'-', letter suffixes,
    wrong ordinal suffixes ("38nd"), "1/2" fractions
  * state names as codes, full names or Indic script

Implementation: vectorised polars expressions; Indic romanisation (a pure
Unicode mapping) runs in Python once per distinct Indic word and is mapped
back in one vectorised replace, so ~20M records normalise in minutes.
No external data or services are used; all word lists are hand-written.

Entry points: ``add_views(df)`` and the CLI ``python -m src.normalize``.
"""
import re
from functools import lru_cache

import polars as pl

# --------------------------------------------------------------------------
# 1. Indic -> Latin romanisation (pure Unicode mapping, cached per unique word)
# --------------------------------------------------------------------------
# All major Indic scripts share Devanagari's code-point layout (same offset in
# a 0x80 block), so one table covers Devanagari, Bengali, Gurmukhi, Gujarati,
# Oriya, Tamil, Telugu, Kannada and Malayalam. Markers model the inherent vowel:
#   \x01 consonant carrying inherent 'a'   \x02 vowel sign (replaces the 'a')
#   \x03 virama (removes the 'a')          \x05 nasal/visarga (keeps the 'a')
_C, _V, _K, _N = "\x01", "\x02", "\x03", "\x05"
_DEV_LETTERS = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
    0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
    0x13: "o", 0x14: "au",
}
_DEV_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
    0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_DEV_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri",
               0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o",
               0x4A: "o", 0x4B: "o", 0x4C: "au"}
_DEV_SIGNS = {0x01: _N + "n", 0x02: _N + "n", 0x03: _N + "h", 0x4D: _K, 0x3C: "",
              0x57: "", 0x70: "n", 0x71: "", 0x4E: "t", 0x64: " ", 0x65: " ", 0x50: "om"}
_NUKTA = {"\u0915\u093c": "\u0958", "\u0916\u093c": "\u0959", "\u0917\u093c": "\u095a",
          "\u091c\u093c": "\u095b", "\u0921\u093c": "\u095c", "\u0922\u093c": "\u095d",
          "\u092b\u093c": "\u095e", "\u092f\u093c": "\u095f"}
_BRAHMIC_BASES = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
INDIC_CLASS = r"[\x{0900}-\x{0D7F}]"
ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00ad"]


def _build_romanisation_table() -> dict:
    """str.translate table: every Brahmic code point -> Latin (with markers)."""
    table = {}
    for base in _BRAHMIC_BASES:
        for off in range(0x80):
            cp = base + off
            if off in _DEV_CONSONANTS:
                table[cp] = _DEV_CONSONANTS[off] + _C
            elif off in _DEV_LETTERS:
                table[cp] = _DEV_LETTERS[off]
            elif off in _DEV_MATRAS:
                table[cp] = _V + _DEV_MATRAS[off]
            elif off in _DEV_SIGNS:
                table[cp] = _DEV_SIGNS[off]
            elif 0x66 <= off <= 0x6F:
                table[cp] = str(off - 0x66)
    table.update({0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l",
                  0x0D7E: "l", 0x0D7F: "k"})               # Malayalam chillu letters
    table.update({ord(z): "" for z in ZERO_WIDTH})
    return table


_ROM_TABLE = _build_romanisation_table()
_CONS = "[b-df-hj-np-tv-z]"
# Hindi schwa deletion: inherent 'a' in VC_CV position and at word end is silent
_MEDIAL_SCHWA = re.compile(rf"([aeiou]{_CONS}{{1,3}})\x01({_CONS}{{1,3}}[aeiou])")
_FINAL_SCHWA = re.compile(r"\x01(?=[^a-z\x01-\x05]|$)")
_MARKERS = re.compile(r"[\x01-\x05]")
_INDIC_RUN = re.compile(r"[\u0900-\u0D7F\u200b-\u200d\u2060\ufeff\u00ad]+")


@lru_cache(maxsize=None)
def romanize_word(word: str) -> str:
    """Romanise one run of Indic characters.

    'लिमिटेड' -> 'limited', 'इन्वेस्टमेंट' -> 'investment', 'బెస్ట్' -> 'best'.
    """
    for combo, single in _NUKTA.items():
        word = word.replace(combo, single)
    s = word.translate(_ROM_TABLE)
    s = s.replace(_C + _V, "").replace(_C + _K, "").replace(_C + _N, "a")
    s = _MEDIAL_SCHWA.sub(r"\1\2", _MEDIAL_SCHWA.sub(r"\1\2", s))
    s = _FINAL_SCHWA.sub("", s).replace(_C, "a")
    return _MARKERS.sub("", s)


def romanize_text(text: str) -> str:
    """Romanise every Indic run inside a string; other characters are untouched."""
    return _INDIC_RUN.sub(lambda m: romanize_word(m.group()), text)


def fold_column(df: pl.DataFrame, col: str, out: str) -> pl.DataFrame:
    """Add ``out`` = romanised, accent-stripped, lowercased copy of ``col``.

    Romanisation runs in Python only on the distinct strings that contain Indic
    characters (cached per word) and is mapped back with a vectorised replace.
    """
    base = pl.col(col).fill_null("")
    uniq = (df.select(base.alias("s")).filter(pl.col("s").str.contains(INDIC_CLASS))
            .unique().get_column("s").to_list())
    if uniq:
        base = base.replace({u: romanize_text(u) for u in uniq})
    folded = (base.str.normalize("NFKD").str.replace_all(r"\p{M}", "")
              .str.replace_many(["ß", "æ", "œ", "ø", "đ", "ł", "ı", "’", "‘"],
                                ["ss", "ae", "oe", "o", "d", "l", "i", "'", "'"])
              .str.to_lowercase().str.strip_chars())
    return df.with_columns(folded.alias(out))


# --------------------------------------------------------------------------
# 2. Hand-written vocabularies (general domain knowledge, no external data)
# --------------------------------------------------------------------------
# Legal forms / professional designations -> canonical short form. Includes
# romanised Indic forms and typo'd variants seen in the data.
LEGAL_CANON = {
    "private": "pvt", "pvt": "pvt", "pvte": "pvt", "prvt": "pvt", "prv": "pvt",
    "prvhte": "pvt", "praivet": "pvt", "praivat": "pvt", "piraivet": "pvt", "pra": "pvt",
    "limited": "ltd", "ltd": "ltd", "limitad": "ltd", "limitd": "ltd", "limitet": "ltd", "li": "ltd",
    "llp": "llp", "elelpi": "llp", "elaelpi": "llp", "opc": "opc",
    "llc": "llc", "lc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp",
    "corporation": "corp", "co": "co", "company": "co", "kampani": "co", "plc": "plc",
    "lp": "lp", "pllc": "pllc", "pc": "pc", "pa": "pa", "esq": "esq", "dc": "dc",
    "dpm": "dpm", "md": "md", "dds": "dds", "cpa": "cpa",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci",
    "eurl": "eurl", "snc": "snc", "scp": "scp", "selarl": "selarl", "scop": "scop",
}
LEGAL_FORMS = sorted(set(LEGAL_CANON.values()))
NAME_STOPWORDS = ["and", "the", "of", "et", "de", "du", "des", "la", "le", "les", "a"]

ADDRESS_CANON = {
    "road": "rd", "street": "st", "str": "st", "avenue": "ave", "av": "ave",
    "lane": "ln", "drive": "dr", "court": "ct", "terrace": "ter", "trail": "trl",
    "boulevard": "blvd", "bd": "blvd", "bvd": "blvd", "highway": "hwy", "place": "pl",
    "circle": "cir", "parkway": "pkwy", "square": "sq", "expressway": "expy",
    "mount": "mt", "mtt": "mt", "fort": "ft", "ftt": "ft", "saint": "st",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "township": "twp", "twnship": "twp", "county": "cnty", "cty": "cnty", "route": "rte",
    "nagar": "ngr", "ngar": "ngr", "colony": "col", "sector": "sec", "phase": "ph",
    "cross": "crs", "layout": "lyt", "extension": "extn", "ext": "extn",
    "rue": "rue", "r": "rue", "allee": "all", "impasse": "imp", "chemin": "ch",
    "bengaluru": "bangalore", "gurugram": "gurgaon", "bombay": "mumbai",
}
# Designators that only announce a number: "#14", "Plot No 14", "H.no 14" -> "14"
ADDRESS_DESIGNATORS = [
    "no", "nos", "h", "hno", "house", "door", "dno", "plot", "flat", "unit", "apt",
    "apartment", "suite", "ste", "office", "level", "floor", "fl", "bldg", "building",
    "shop", "po", "box", "near", "opp", "behind", "c", "o", "s", "at",
]
ADDRESS_NULLS = r"<\s*null\s*>|\bnull\b|\bn\s*/\s*a\b|\bnone\b|\bnil\b"

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl",
    "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in",
    "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
IN_STATES = {  # includes romanised native-script spellings produced by romanize()
    "andhra pradesh": "ap", "andhr pradesh": "ap", "arunachal pradesh": "ar",
    "assam": "as", "bihar": "br", "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "gujrat": "gj",
    "haryana": "hr", "hariyana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "karnatak": "ka", "kerala": "kl", "keral": "kl",
    "madhya pradesh": "mp", "madhy pradesh": "mp", "maharashtra": "mh",
    "maharashtr": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz",
    "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb", "panjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "tg", "telangan": "tg", "ts": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "pashchim bangal": "wb", "delhi": "dl", "dilli": "dl",
    "nct of delhi": "dl", "jammu and kashmir": "jk", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py",
}
_STATE_LOOKUP = {**{c: c for c in US_STATES.values()}, **US_STATES,
                 **{c: c for c in IN_STATES.values()}, **IN_STATES}

DBA_MARKERS = (r"(?:\bd\s*/\s*b\s*/\s*a\b|\bdba\b|\bf\s*/\s*k\s*/\s*a\b|\bfka\b|"
               r"\ba\s*/\s*k\s*/\s*a\b|\baka\b|\btrading as\b|\bdoing business as\b|"
               r"\bformerly known as\b)")
TLD = r"\.(?:co\.in|org\.in|net\.in|com|net|org|in|co|fr|biz|info|us|io)\b"


# --------------------------------------------------------------------------
# 3. Name views
# --------------------------------------------------------------------------
def _name_cleanup(e: pl.Expr) -> pl.Expr:
    """Punctuation/junk handling on an already folded name string."""
    e = (e.str.replace_all(r"#\s*\d+", " ")                  # ids: "#66553"
         .str.replace_all(r"^\s*m\s*/\s*s\.?\s+", " ")        # "M/s " prefix
         .str.replace_all(r"www\.", " ").str.replace_all(TLD, " ")
         .str.replace_all(r"['`]", "")                        # fanni's -> fannis
         .str.replace_all(r"\.", "")                          # l.l.c. -> llc
         .str.replace_all(r"[&+]", " and "))
    for d, letter in (("1", "l"), ("0", "o"), ("5", "s")):    # c1ark, 5atpura
        e = (e.str.replace_all(rf"([a-z]){d}([a-z])", "${1}" + letter + "${2}")
             .str.replace_all(rf"\b{d}([a-z]{{3,}})", letter + "${1}"))
    return e.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()


def _tokens(e: pl.Expr) -> pl.Expr:
    """Whitespace split, drop empties, de-duplicate keeping order."""
    return (e.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
            .list.unique(maintain_order=True))


def _skeleton(e: pl.Expr) -> pl.Expr:
    """Rough phonetic key: c/q->k, ph->f, z->s, drop inner vowels, collapse doubles."""
    e = (e.str.replace_all("ph", "f", literal=True).str.replace_all(r"[cq]", "k")
         .str.replace_all("z", "s", literal=True).str.replace_all(r"\B[aeiouyhw]+", ""))
    doubles = [c * 2 for c in "bdfgklmnprstv"]
    return e.str.replace_many(doubles, [d[0] for d in doubles])


def _name_frame(df: pl.DataFrame, name_col: str) -> pl.DataFrame:
    df = fold_column(df, name_col, "_nf")
    f = pl.col("_nf")
    has_dba = f.str.contains(DBA_MARKERS)
    has_pipe = f.str.contains(r"\|")
    # In this data the real name follows the DBA/FKA marker; the prefix is invented.
    main = (pl.when(has_dba).then(f.str.replace(rf"^.*?{DBA_MARKERS}", ""))
            .when(has_pipe).then(f.str.extract(r"^(.*?)\|", 1)).otherwise(f))
    alt = (pl.when(has_dba).then(f.str.extract(rf"^(.*?){DBA_MARKERS}", 1))
           .when(has_pipe).then(f.str.extract(r"\|(.*)$", 1)).otherwise(pl.lit("")))
    out = df.with_columns(
        _tokens(_name_cleanup(main)).list.eval(pl.element().replace(LEGAL_CANON))
        .alias("name_tokens"),
        _tokens(_name_cleanup(alt.fill_null(""))).list.eval(pl.element().replace(LEGAL_CANON))
        .alias("_alt_tokens"),
        has_dba.alias("name_has_dba"),
        f.str.contains(r"^[^\w]*[@#][\w.]+\s*$").alias("name_is_handle"),
        f.str.contains(rf"www\.|{TLD}").alias("name_is_domain"),
        pl.col(name_col).fill_null("").str.contains(INDIC_CLASS).alias("name_has_indic"),
    )
    not_legal = ~pl.element().is_in(LEGAL_FORMS)
    content = not_legal & ~pl.element().is_in(NAME_STOPWORDS)
    out = out.with_columns(
        pl.col("name_tokens").list.eval(pl.element().filter(not_legal)).alias("name_core_tokens"),
        pl.col("name_tokens").list.eval(pl.element().filter(pl.element().is_in(LEGAL_FORMS)))
        .list.unique().list.sort().list.join(" ").alias("name_legal"),
        pl.col("_alt_tokens").list.eval(pl.element().filter(not_legal)).list.join(" ")
        .alias("name_alt_core"),
        pl.col("name_tokens").list.eval(pl.element().filter(content)).alias("_content"),
    )
    return out.with_columns(
        pl.col("name_core_tokens").list.join(" ").alias("name_core"),
        pl.col("_content").list.join("").alias("name_concat"),
        pl.col("_content").list.eval(pl.element().str.slice(0, 1)).list.join("")
        .alias("name_initials"),
        _skeleton(pl.col("_content").list.join(" ")).alias("name_skeleton"),
    ).drop("_alt_tokens", "_content", "_nf")


# --------------------------------------------------------------------------
# 4. Address views
# --------------------------------------------------------------------------
_MULTIWORD_STATES = {k: v for k, v in {**US_STATES, **IN_STATES}.items() if " " in k}
_ADDR_TOKEN_MAP = {**ADDRESS_CANON,
                   **{k: v for k, v in _STATE_LOOKUP.items() if " " not in k and len(k) > 2}}


def _address_frame(df: pl.DataFrame, addr_col: str) -> pl.DataFrame:
    df = fold_column(df, addr_col, "_af")
    f = pl.col("_af").str.replace_all(ADDRESS_NULLS, " ")
    # states: every comma-separated component that is exactly a state name or code
    state = (f.str.split(",")
             .list.eval(pl.element().str.replace_all(r"[^a-z ]", "").str.strip_chars()
                        .replace_strict(_STATE_LOOKUP, default=None))
             .list.drop_nulls().list.unique(maintain_order=True))
    f = f.str.replace_many([" first ", " second ", " third ", " fourth ", " fifth ", " sixth ",
                            " seventh ", " eighth ", " ninth ", " tenth "],
                           [" 1 ", " 2 ", " 3 ", " 4 ", " 5 ", " 6 ", " 7 ", " 8 ", " 9 ", " 10 "])
    e = (f.str.replace_all(r"(\d)(?:st|nd|rd|th)\b", "${1}")      # 38nd -> 38
         .str.replace_all(r"\b(\d+)\s*1\s*/\s*2\b", "${1}")        # 77 1/2 -> 77
         .str.replace_all(r"(\d)([a-z])\b", "${1} ${2}")           # 33a -> 33 a
         .str.replace_all(r"['`]", "").str.replace_all(".", " ", literal=True)
         .str.replace_all(r"\b([a-z]+)(\d)", "${1} ${2}"))           # no2 -> no 2
    for full, code in _MULTIWORD_STATES.items():                   # "north carolina"
        e = e.str.replace_all(rf"\b{full}\b", code)
    toks = _tokens(e.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
    is_num = pl.element().str.contains(r"^\d+$")
    out = df.with_columns(
        toks.list.eval(pl.element().filter(~is_num).replace(_ADDR_TOKEN_MAP))
        .list.eval(pl.element().filter((pl.element().str.len_chars() > 1)
                                       & ~pl.element().is_in(ADDRESS_DESIGNATORS)))
        .list.unique(maintain_order=True).alias("addr_tokens"),
        toks.list.eval(pl.element().filter(is_num).str.strip_chars_start("0"))
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.unique(maintain_order=True).alias("addr_numbers"),
        state.alias("addr_states"),
        pl.col(addr_col).fill_null("").str.contains(INDIC_CLASS).alias("addr_has_indic"),
    )
    return out.with_columns(
        pl.col("addr_tokens").list.join(" ").alias("addr_clean"),
        (pl.col("addr_tokens").list.len() + pl.col("addr_numbers").list.len() == 0)
        .alias("addr_empty"),
    ).drop("_af")


VIEW_COLUMNS = [
    "name_tokens", "name_core_tokens", "name_core", "name_alt_core", "name_concat",
    "name_initials", "name_skeleton", "name_legal", "name_has_dba", "name_is_handle",
    "name_is_domain", "name_has_indic", "addr_tokens", "addr_numbers", "addr_clean",
    "addr_states", "addr_empty", "addr_has_indic",
]


def add_views(df: pl.DataFrame, name_col: str = "business_name",
              addr_col: str = "business_address") -> pl.DataFrame:
    """Add all normalized views (VIEW_COLUMNS) to a source DataFrame."""
    return _address_frame(_name_frame(df, name_col), addr_col)


# --------------------------------------------------------------------------
# 5. CLI: normalise every source file once, cache as parquet
# --------------------------------------------------------------------------
def main():
    """Normalise train/test S1-S3 and write WORK_DIR/normalized/<split>_<src>.parquet."""
    import argparse
    import time
    from pathlib import Path

    from src.config import DATA_DIR, WORK_DIR, test_paths, train_paths
    from src.io_utils import read_source

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out-dir", default=str(Path(WORK_DIR) / "normalized"))
    ap.add_argument("--max-rows", type=int, default=None, help="cap rows per file (testing)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(f"train_{k}", p) for k, p in train_paths(args.data_dir).items() if k != "gt"]
    jobs += [(f"test_{k}", p) for k, p in test_paths(args.data_dir).items()]
    for name, path in jobs:
        t = time.time()
        df = add_views(read_source(path, args.max_rows))
        df.write_parquet(out / f"{name}.parquet")
        print(f"{name:10s} {df.height:>10,} rows {time.time() - t:7.1f}s", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
