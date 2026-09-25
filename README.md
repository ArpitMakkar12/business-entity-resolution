# Business Entity Resolution — Amazon ML Challenge 2026

Matches each Source 1 business record to its Source 2 / Source 3 records.
Pipeline: normalization → multi-pass blocking → LightGBM pair classifier →
F0.5-aware set decision. (This README is expanded as stages are added.)

## Layout
```
src/
  config.py     paths (overridable via BER_DATA_DIR / BER_WORK_DIR / BER_OUTPUT_DIR)
  io_utils.py   TSV readers and submission writers
  eda.py        exploratory analysis report
  normalize.py  canonical name/address views (romanisation, legal forms, numbers, states)
tests/
  test_normalize.py  regression tests on real noise patterns
notebooks/
  00_colab_bootstrap.ipynb   Colab setup: data, code, dependencies, runs
```

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```
Expected data layout (the challenge's `dataset/` folder):
```
<BER_DATA_DIR>/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
<BER_DATA_DIR>/test/test_source{1,2,3}.tsv
```

## Run
```bash
python -m src.eda --out-dir work/eda                          # full data
python -m src.eda --out-dir work/eda_sample --max-rows 300000 # quick sample
python -m src.normalize                    # -> work/normalized/{train,test}_s{1,2,3}.parquet
python -m tests.test_normalize             # regression tests
```

## Compliance
Uses only the provided training/test files. No external databases, APIs,
geocoding or internet lookups at any stage; after installing dependencies the
pipeline runs fully offline. Models used are MIT/Apache-2.0 licensed and ≤8B
parameters (listed in the methodology document).
