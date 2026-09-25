"""Regression tests for src/normalize.py, built from real noise patterns.

Run from the project root:  python -m tests.test_normalize   (or: python -m pytest -q tests)
"""
import polars as pl

from src.normalize import add_views, romanize_text


def _views(names, addrs=None):
    addrs = addrs or [""] * len(names)
    df = pl.DataFrame({"entity_id": [f"S2-{i}" for i in range(len(names))],
                       "business_name": names, "business_address": addrs,
                       "country": ["US"] * len(names)})
    return add_views(df)


def test_romanisation():
    assert romanize_text("लिमिटेड") == "limited"
    assert romanize_text("इन्वेस्टमेंट") == "investment"
    assert romanize_text("బెస్ట్") == "best"
    assert romanize_text("Sun पावर Provision") == "Sun pavar Provision"


def test_name_core_collapses_noise():
    v = _views(["Raj Healthcare Private Limited", "PRVHTE RAJ HEALTHCARE [LIMITED]",
                "M/s RAJ HEALTHCARE PRIVATE LIMITED", "Fluxsol F/K/A Raj Healthcare Private Limited",
                "C1ark, Livingston And Kendall Inc", ">> Cohen American Series (LLC)",
                "Anna Cataldo, Esq., P.C. #66553", "HÁRBOR-FEDERATION"])
    assert v["name_core"].to_list()[:4] == ["raj healthcare"] * 4
    assert v["name_core"][4] == "clark livingston and kendall"
    assert v["name_core"][5] == "cohen american series"
    assert v["name_core"][6] == "anna cataldo"
    assert v["name_core"][7] == "harbor federation"
    assert v["name_alt_core"][3] == "fluxsol"
    assert v["name_legal"][0] == "ltd pvt"


def test_handles_and_domains():
    v = _views(["@alikadavis", "navaportillo.com", "Nava and Portillo LLC"])
    assert v["name_is_handle"][0] and v["name_is_domain"][1]
    assert v["name_concat"][1] == v["name_concat"][2] == "navaportillo"


def test_address_views():
    v = _views(["a", "b", "c", "d"],
               ["6326 Traveler Lane, West Jordan City, UT",
                "West Jordan City, Utah, 06326 Traveler Ln",
                "Level No.g-2, <NULL>, Hyderabad, TG",
                "L-1/24., Sri Krishna Puri, Phulwari, Patna, बिहार"])
    assert v["addr_clean"][0] == "traveler ln jordan city ut"
    assert sorted(v["addr_clean"][1].split()) == sorted(v["addr_clean"][0].split())
    assert v["addr_numbers"][0].to_list() == v["addr_numbers"][1].to_list() == ["6326"]
    assert v["addr_numbers"][2].to_list() == ["2"] and "null" not in v["addr_clean"][2]
    assert v["addr_states"][3].to_list() == ["br"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
