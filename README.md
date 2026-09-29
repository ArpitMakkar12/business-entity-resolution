# Business Entity Resolution — Amazon ML Challenge 2026

**Team Data Whisperers** (VIT-AP University): Keshav Sharma, Akash Pandey, Syed Wahid, Arpit Makkar

A complete entity-resolution pipeline built in the 72-hour Amazon ML Challenge 2026
(25–27 September 2026). For every business record in Source 1 it finds all records of
the same business in Source 2 and Source 3, across roughly 24 million noisy,
multilingual records from India, the US and France. It uses only the provided data, runs
fully offline and uses no pretrained models.

| | |
|---|---|
| **Final public leaderboard score** (macro F0.5) | **0.962332**, up from 0.7186 on the first submission, over 17 submissions |
| Validation macro F0.5 (100,000 held-out entities) | 0.9773 (model A) and 0.9782 (model B) |
| Test candidate pairs | 53.8 M for 1.73 M entities (31.1 per entity, reduction ratio 0.999992) |
| Models | two LightGBM classifiers (MIT licence) over 61 hand-built features |
| Runtime | about 3.5 h end to end on one 12-core Colab machine, no GPU |

---

## The problem

Each business appears once, cleanly, in Source 1 and several times, noisily, in
Sources 2 and 3:

- typos, reordered words and names moved into DBA/"f/k/a" aliases
- legal forms moved, duplicated or changed (Pvt Ltd / Private Limited / LLP, SARL / SAS)
- about 23% of India names written in Devanagari, Kannada, Telugu, Tamil, Gujarati or
  Bengali script
- partial, reordered or empty addresses, and house numbers that are dropped or zero-padded
- France, which appears only in the test set

The hardest part is the **look-alike "sibling" businesses** in the test set: the same
street, a neighbouring house number, and a name that differs by one word or its legal
form. The metric is macro F0.5 per Source 1 entity, which weights precision twice as much
as recall, so every wrong merge is expensive.

---

## Approach

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
 6. decision       house-number conflict rule, exclusivity-aware joint probability,
                   one record -> one S1, threshold 0.7
      │                                              -> matching_results.tsv
```

### Key ideas

1. **Measure blocking before tuning anything.** `src/block_eval.py` measures candidate
   recall against the *full* training pool. It exposed a polars 1.35 bug that mis-read
   list columns of sliced frames and had built blocking keys from the wrong rows. Fixing
   it took the score from 0.72 to 0.93.
2. **Synthetic sibling negatives** (`src/siblings.py`). The training data rarely contains
   the look-alike businesses of the test set. So we made them from the training data:
   copies of true matches with a shifted house number and a changed legal form or filler
   word, added to the training pool as non-matches for 90% of entities.
3. **Rules that target the test distractors.** A house-number conflict rule, and an
   **exclusivity-aware joint probability** (`src/assign.py`): a record can belong to at
   most one entity, so a record that fits two entities about equally well is predicted for
   neither.
4. **Reverse search** in blocking (preset `v17`). Every S2/S3 record also proposes its best
   Source 1 entities. This recovered true copies that had been crowded out of their own
   entity's candidate list by look-alike records.
5. **Averaging two models** trained on different samples.

### What moved the leaderboard

| Change | Effect |
|---|---|
| Blocking rebuilt and checked against the full training pool (polars bug fixed) | 0.7186 → 0.9256 |
| House-number conflict rule | +0.022 |
| Synthetic sibling negatives (40% of entities, then 90%) | +0.006, then +0.0018 |
| Averaging two models | +0.0002 |
| Exclusivity-aware joint probability | +0.0003 |
| Reverse search in blocking | +0.0012 |

**What did not transfer to the leaderboard**, although some of it helped on validation:
group-consistency re-scoring (`src/stage2.py`), more training data alone, stricter
thresholds, stricter conflict rules, and stripping honorifics such as "Shri" or "Dr"
(`src/patch_names.py`).

### Lessons learned

- **Validation must reproduce the test set's distractors.** Until our training pool held
  about as many look-alike businesses per entity as the test pool, validation overestimated
  the leaderboard by 0.05.
- **Check every change on validation before spending a submission on it.** Two tempting
  ideas (reassigning contested records by name similarity, and stripping honorifics) looked
  right on examples but were worse when measured.
- **Leaderboard probes can locate the problem.** Blanking one country per submission showed
  that most of the remaining gap was in India, and was not in the parts of the pipeline we
  were still tuning.

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Versions are pinned in `requirements.txt` (polars 1.44.2, rapidfuzz 3.14.6,
lightgbm 4.7.0). **Do not use polars 1.35.x.**

**Data.** The challenge data belongs to the organisers and is not part of this
repository. The code expects the challenge's `dataset/` folder:

```
<BER_DATA_DIR>/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
<BER_DATA_DIR>/test/test_source{1,2,3}.tsv
```

Paths are set with environment variables: `BER_DATA_DIR` (default `./dataset`),
`BER_WORK_DIR` (intermediate files, default `./work`) and `BER_OUTPUT_DIR`. `BER_N_JOBS`
sets the number of threads. On Google Colab, `notebooks/00_colab_bootstrap.ipynb` mounts
Drive, copies the data, fetches the code and installs the dependencies.

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
       --post conflict_p90 --assign joint --out-dir output
```

The result is `output/matching_results.tsv` (5,588,460 predicted matches; 6.07% of
entities without a match) and `output/candidate_pairs.tsv`. Cached stages
(`test_cands.parquet`, `test_scored.parquet`, `train_feats.parquet`, ...) are reused on
re-runs; add `--force` to recompute them. To check the files with the organisers'
validator, add `--validator tools/validate_submission.py`.

### Useful `src.predict` options

| Option | Effect |
|---|---|
| `--blend DIR [DIR ...]` | average the test probabilities of other model folders (same `test_cands.parquet`) |
| `--blend-weights w0 w1 ...` | weighted instead of equal average (main model first) |
| `--post VARIANT` | house-number rule from `src/postprocess.py` (`conflict_p90` is the final one) |
| `--assign joint` | exclusivity-aware joint probability (`strength` / `explain` also exist) |
| `--tau T`, `--tau-country India=0.6 ...` | global / per-country threshold |
| `--rescore` | recompute features and scores, reuse candidates |
| `--blank-country C` | leaderboard probe: predict no matches for one country |

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
                     assignment -> decision -> output files (+ optional validator)
  decision.py        exclusivity, threshold / expected-F0.5 rules, official metric (macro F0.5)
  postprocess.py     house-number conflict rules against sibling-business false merges
  assign.py          exclusivity-aware joint probability; owner reassignment variants

  # evaluation and diagnostics
  block_eval.py      blocking recall / cost of configurations on the full training pool
  error_analysis.py  where validation F0.5 is lost, with examples
  tune_post.py       validation cost and test effect of each post-processing rule
  eval_joint.py      joint probability on validation with realistic competition
  eval_blend.py      blend weight / threshold / rules on entities held out for both models
  audit.py           test-vs-validation prediction audit with examples
  audit_country.py   per-country examples of odd matches, misses and near misses
  diag_s1.py         look-alike entities inside Source 1, train vs test
  diag_k.py          test vs validation by type of predicted set
  diag_contest.py    records claimed by several entities on test
  diag_shift.py      adversarial check: do test predictions differ from validation ones?
  diag_unclaimed.py  probability bands and unclaimed test records
  diag_tokens.py     name words more frequent on test than on train

  # tried, not used in the final submission
  stage2.py          group-consistency re-scoring
  patch_names.py     honorific stripping

tests/
  test_normalize.py  regression tests on real noise patterns
  make_synthetic.py  small synthetic dataset for smoke tests (with sibling distractors)
  eval_synthetic.py  scores a synthetic run against its hidden test truth
tools/               organisers' submission validator (copied from the challenge resources)
notebooks/
  00_colab_bootstrap.ipynb  Colab setup: Drive, data, code, dependencies
Documentation_template.md   methodology write-up submitted with the solution
```

Smoke test on synthetic data (seconds, no challenge data needed):

```bash
python -m tests.make_synthetic --out /tmp/synth
export BER_DATA_DIR=/tmp/synth BER_WORK_DIR=/tmp/synth_work
python -m src.normalize && python -m tests.test_normalize
python -m src.train --n-train 3000 --n-valid 1500 --blocking v16 --siblings 0.9
python -m src.predict --post conflict_p90 --assign joint
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
| 16 | 13 + exclusivity-aware joint probability | 0.9611 |
| **17** | **16 with reverse search in both models (blocking v17)** | **0.9623** |

---

## Compliance

- Only the provided training and test files are used. No external databases, APIs,
  geocoding, gazetteers or internet lookups at any stage; after `pip install` the pipeline
  runs fully offline.
- Normalisation vocabularies (legal forms, street types, state names) are short
  hand-written lists in the code. Aliases and synthetic siblings are derived from the
  training data only.
- No pretrained or large language models. Libraries: LightGBM (MIT), polars (MIT),
  rapidfuzz (MIT), NumPy (BSD), psutil (BSD), tqdm (MIT / MPL-2.0).
- `.gitignore` keeps the challenge data and all intermediate files out of the repository.

## Acknowledgements

Thanks to the Amazon ML Challenge 2026 organisers and Unstop for the problem, the data
and the evaluation platform.
