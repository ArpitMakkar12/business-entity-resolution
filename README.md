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
  aliases.py    romanised-token -> English-token aliases learned from training pairs
  blocking.py   multi-key inverted index (1/block-size weights) + similarity re-rank
  block_eval.py blocking recall/cost comparison of configurations on training data
  features.py   pair features (rapidfuzz similarities, token/IDF overlap, address, context)
  decision.py   exclusivity, expected-F0.5 set choice, official metric (macro F0.5)
  train.py      end-to-end training + validation report
  predict.py    test prediction -> output/matching_results.tsv + candidate_pairs.tsv
  audit.py      test-vs-validation prediction audit with examples (no labels needed)
  postprocess.py house-number conflict rules against sibling-business false merges
  tune_post.py  validation cost / test effect of each post-processing rule
tests/
  test_normalize.py  regression tests on real noise patterns
  make_synthetic.py  small synthetic dataset for pipeline smoke tests
tools/
  validate_submission.py  organisers' validator (unchanged copy)
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
python -m src.block_eval                   # blocking recall report
python -m src.train                        # -> work/model/ (model.txt, params.json, report.json)
python -m src.predict --validator tools/validate_submission.py   # -> work/output/*.tsv
```

Smoke test on synthetic data (a few seconds):
```bash
python -m tests.make_synthetic --out /tmp/synth
export BER_DATA_DIR=/tmp/synth BER_WORK_DIR=/tmp/synth_work
python -m src.normalize && python -m src.train --n-train 3000 --n-valid 1500 && python -m src.predict
```

## Compliance
Uses only the provided training/test files. No external databases, APIs,
geocoding or internet lookups at any stage; after installing dependencies the
pipeline runs fully offline. Models used are MIT/Apache-2.0 licensed and ≤8B
parameters (listed in the methodology document).
