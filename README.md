# Business Entity Resolution — Amazon ML Challenge 2026

**Team Data Whisperers** (VIT-AP University): Keshav Sharma, Akash Pandey, Syed Wahid, Arpit Makkar

For every Source 1 (S1) business record, find all records of the same business in
Source 2 and Source 3 (S2/S3). The records are noisy (typos, reordered and
transliterated names, moved legal forms, Indic scripts, partial addresses), and
the test set is full of look-alike "sibling" businesses: the same street, a
neighbouring house number and a name that differs in one word.

| | |
|---|---|
| **Final public leaderboard score (macro F0.5)** | **0.962332** (submission 17, from 0.7186 at submission 1) |
| Validation macro F0.5 (100,000 held-out S1 entities) | 0.9773 (model A), 0.9782 (model B) |
| Test candidate pairs | 53,832,938 (31.1 per S1 entity, reduction ratio 0.999992) |
| Data used | only the provided training and test files, fully offline |
| Models | two LightGBM classifiers (MIT licence), no pretrained models |

The methodology write-up is in `Documentation_template.md`.

---

## Pipeline

```
 S1, S2, S3 (.tsv)
      │
      ▼
 1. normalize      canonical name / address views: Unicode romanisation of 9 Indic
                   scripts, legal forms, street types, states, house numbers
      │
      ▼
 2. aliases        472 romanised-word -> English-word aliases learned from training pairs
      │
      ▼
 3. blocking       multi-key inverted index per country (9 key types),
                   1/block-size scoring, union re-rank, name-only retrieval,
                   reverse search (each S2/S3 record also proposes its best S1)
      │                                              -> candidate_pairs.tsv
      ▼
 4. features       61 pair features: name / address similarities, IDF overlap,
                   house-number agreement and distance, candidate-group context,
                   name ambiguity
      │
      ▼
 5. LightGBM x 2   model A (300k training entities) + model B (700k), both trained
                   with synthetic sibling negatives; probabilities averaged
      │
      ▼
 6. decision       house-number conflict rule (p < 0.90), exclusivity-aware joint
                   probability, one record -> one S1, threshold 0.7
      │                                              -> matching_results.tsv
      ▼
 official validator (tools/validate_submission.py)
```

### What made the difference

| Step | Leaderboard effect |
|---|---|
| Blocking rebuilt and verified against the full training pool (it also exposed a polars 1.35 list-slicing bug that had built keys from the wrong rows) | 0.7186 → 0.9256 |
| House-number conflict rule against sibling businesses | +0.022 |
| **Synthetic sibling negatives** (`src/siblings.py`): copies of true matches with a shifted house number and a changed legal form or filler word, injected into the training pool as non-matches | +0.006, and +0.0018 more when raised from 40% to 90% of entities |
| Averaging two models trained on different samples | +0.0002 |
| **Joint (exclusivity-aware) probability** (`src/assign.py`): a record that fits two S1 entities about equally is not predicted for either | +0.0003 |
| **Reverse search** (blocking preset `v17`): recovers true copies crowded out of their entity's candidate list | +0.0012 |

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Versions are pinned in `requirements.txt` (polars 1.44.2, rapidfuzz 3.14.6,
lightgbm 4.7.0). **Do not use polars 1.35.x**: it mis-reads list columns of sliced
frames.

Data layout (the challenge's `dataset/` folder):

```
<BER_DATA_DIR>/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
<BER_DATA_DIR>/test/test_source{1,2,3}.tsv
```

All paths are set through environment variables: `BER_DATA_DIR` (default
`./dataset`), `BER_WORK_DIR` (intermediate files, default `./work`) and
`BER_OUTPUT_DIR`. `BER_N_JOBS` sets the number of threads. On Google Colab,
`notebooks/00_colab_bootstrap.ipynb` mounts Drive, copies the data, fetches the
code and installs the dependencies.

---

## Reproduce the final submission (0.962332)

Timings are on a Colab L4 High-RAM runtime (12 vCPU, 53 GB RAM); no GPU is used.
Seeds are fixed (`SEED = 42`).

```bash
export BER_DATA_DIR=/path/to/dataset BER_WORK_DIR=work

# 1. normalize all six source files -> work/normalized/*.parquet            (~5 min)
python -m src.normalize

# 2. train the two models (blocking v16, siblings for 90% of entities)
python -m src.train --blocking v16 --siblings 0.9 --model-dir work/model_v3                    # model A, ~25 min
python -m src.train --blocking v16 --siblings 0.9 --n-train 700000 --model-dir work/model_v4   # model B, ~60 min

# 3. copies of both model folders that use blocking v17 (v16 + reverse search) on test
for m in model_v3 model_v4; do
  mkdir -p work/${m}r && cp work/$m/model.txt work/$m/aliases.json work/${m}r/
  python -c "import json; p=json.load(open('work/$m/params.json')); p['blocking']='v17'; json.dump(p, open('work/${m}r/params.json','w'))"
done

# 4. model A: test blocking (v17) + scoring                                    (~60 min)
python -m src.predict --model-dir work/model_v3r --post conflict_p90 --out-dir work/out_a
# 5. model B: scoring of the same candidates                                   (~48 min)
cp work/model_v3r/test_cands.parquet work/model_v4r/
python -m src.predict --model-dir work/model_v4r --post conflict_p90 --out-dir work/out_b

# 6. final: average of A and B, conflict rule, joint probability, threshold 0.7 (~3 min)
python -m src.predict --model-dir work/model_v3r --blend work/model_v4r \
       --post conflict_p90 --assign joint \
       --out-dir output --validator tools/validate_submission.py
```

The result is `output/matching_results.tsv` (5,588,460 predicted matches; 6.07%
of S1 entities without a match) and `output/candidate_pairs.tsv`. Cached stages
(`test_cands.parquet`, `test_scored.parquet`, `train_feats.parquet`, ...) are
reused on re-runs; add `--force` to recompute them.

### Useful `src.predict` options

| Option | Effect |
|---|---|
| `--blend DIR [DIR ...]` | average the test probabilities of other model folders (same `test_cands.parquet`) |
| `--blend-weights w0 w1 ...` | weighted instead of equal average (main model first) |
| `--post VARIANT` | house-number rule from `src/postprocess.py` (`conflict_p90` is the final one) |
| `--assign joint` | exclusivity-aware joint probability (`strength` / `explain` also exist) |
| `--tau T`, `--tau-country India=0.6 ...` | global / per-country threshold |
| `--rescore` | recompute features and scores, reuse candidates |
| `--blank-country C` | probe only: predict no matches for a country (measures its score on the leaderboard) |

---

## Project layout

```
src/
  # pipeline
  config.py          paths, seed, threads (BER_* environment variables)
  io_utils.py        TSV readers (ids kept as strings) and submission writers
  eda.py             exploratory data analysis report
  normalize.py       canonical name / address views, Indic romanisation (Unicode mapping)
  aliases.py         romanised-token -> English-token aliases learned from training pairs
  blocking.py        multi-key inverted index, 1/block-size scoring, union re-rank,
                     name-only retrieval, reverse search; presets v6 ... v19
  siblings.py        synthetic sibling-business hard negatives (training only)
  features.py        61 pair features (rapidfuzz, polars)
  train.py           aliases -> siblings -> blocking -> features -> LightGBM -> decision tuning
  predict.py         test blocking -> features -> scoring -> blend -> post rule ->
                     assignment -> decision -> output files + official validator
  decision.py        exclusivity, threshold / expected-F0.5 rules, official metric (macro F0.5)
  postprocess.py     house-number conflict rules against sibling-business false merges
  assign.py          joint (exclusivity-aware) probability and owner reassignment of
                     records that fit several S1 entities

  # evaluation and diagnostics (training labels or test predictions only)
  block_eval.py      blocking recall / cost of configurations on the full training pool
  error_analysis.py  where validation F0.5 is lost, with examples
  tune_post.py       validation cost and test effect of each post-processing rule
  eval_joint.py      joint probability on validation with realistic S1 competition
  eval_blend.py      blend weight / threshold / rules on entities held out for both models
  audit.py           test-vs-validation prediction audit with examples
  audit_country.py   per-country examples of odd matches, misses and near misses
  diag_s1.py         look-alike entities inside Source 1, train vs test
  diag_k.py          test vs validation by type of predicted set
  diag_contest.py    records claimed by several S1 entities on test
  diag_shift.py      adversarial check: do test predictions differ from validation ones?
  diag_unclaimed.py  probability bands and unclaimed test records
  diag_tokens.py     name words more frequent on test than on train

  # tried, not used in the final submission
  stage2.py          group-consistency re-scoring (validation +0.0006, no leaderboard gain)
  patch_names.py     honorific stripping ("Shri", "Sri", ...) (validation -0.0001)

tests/
  test_normalize.py  regression tests on real noise patterns
  make_synthetic.py  small synthetic dataset for smoke tests (with sibling distractors)
  eval_synthetic.py  scores a synthetic run against its hidden test truth
tools/
  validate_submission.py  organisers' validator (unchanged copy)
notebooks/
  00_colab_bootstrap.ipynb  Colab setup: Drive, data, code, dependencies
```

Smoke test on synthetic data (seconds):

```bash
python -m tests.make_synthetic --out /tmp/synth
export BER_DATA_DIR=/tmp/synth BER_WORK_DIR=/tmp/synth_work
python -m src.normalize && python -m tests.test_normalize
python -m src.train --n-train 3000 --n-valid 1500 --blocking v16 --siblings 0.9
python -m src.predict --post conflict_p90 --assign joint --validator tools/validate_submission.py
python -m tests.eval_synthetic /tmp/synth/test_truth_hidden.tsv /tmp/synth_work/output/matching_results.tsv
```

---

## Leaderboard history

| # | Change | Public score |
|---|---|---|
| 1 | Baseline (blocking affected by the polars slicing bug) | 0.7186 |
| 2 | Fixed blocking, 1/block-size key weights, union re-rank | 0.9256 |
| 3 | + house-number conflict rule (p < 0.95) | 0.9473 |
| 4 | Retrained with synthetic sibling negatives + number-distance features | 0.9536 |
| 5 | 4 + conflict rule p < 0.90 | 0.9573 |
| 6 | Realistic siblings, triple-name key, name-only slot, ambiguity features | 0.9575 |
| 7 | 6 + conflict rule p < 0.90 | 0.9588 |
| 8 | 7 with threshold 0.8 | ≤ 0.9588 |
| 9 | 7 + stage-2 group re-scoring | ≤ 0.9588 |
| 10 | Model A: blocking v16 (name-only retrieval) + siblings for 90% of entities | 0.9606 |
| 11 | 10 with conflict rule p < 0.95 | ≤ 0.9606 |
| 12 | Model B: 700k training entities | 0.9600 |
| 13 | Average of models A and B | 0.9608 |
| 14–15 | Probes: France / India predicted empty (per-country diagnosis) | 0.829 / 0.579 |
| 16 | 13 + joint (exclusivity-aware) probability | 0.9611 |
| **17** | **16 with reverse search in both models (blocking v17)** | **0.9623** |

What did not transfer to the leaderboard, although some of it helped on
validation: stage-2 re-scoring, more training data alone, stricter thresholds,
stricter conflict rules, stripping honorifics. The lesson from the whole
challenge: validation has to reproduce the test set's distractors, and only
changes aimed at those (blocking recall, sibling negatives, conflict and joint
rules) moved the score.

---

## Compliance

- Only the provided training and test files are used. No external databases,
  APIs, geocoding, gazetteers or internet lookups at any stage; after
  `pip install` the pipeline runs fully offline.
- Normalisation vocabularies (legal forms, street types, state names) are short
  hand-written lists in the code. Aliases and synthetic siblings are derived from
  the training data only.
- No pretrained or large language models. Libraries: LightGBM (MIT), polars (MIT),
  rapidfuzz (MIT), NumPy (BSD), psutil (BSD), tqdm (MIT / MPL-2.0).
- `.gitignore` keeps the challenge data and all intermediate files out of the repository.