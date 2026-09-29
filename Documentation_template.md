# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Data Whisperers  
**Team Members:** Keshav Sharma, Akash Pandey, Syed Wahid, Arpit Makkar (VIT-AP University)  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We built a four-stage pipeline: rule-based normalisation (including Unicode romanisation of Indic scripts), a multi-key inverted-index blocking stage, a LightGBM pair classifier over 61 similarity features, and an F0.5-tuned set decision with one-to-one assignment; the final submission averages two such classifiers trained on different samples, adds a reverse search to blocking and resolves records claimed by several entities with an exclusivity-aware joint probability. The main innovation is **synthetic "sibling business" hard negatives** generated from the training data only. These are near-copies of real matches with a neighbouring house number and a changed legal form or filler word, and they teach the model to reject the look-alike distractor businesses that dominate the test set. Validation macro F0.5 is **0.977–0.978** (Section 5), and the public leaderboard score rose from 0.719 to **0.962** (0.962332) across 17 submissions.

---

## 2. Methodology

### 2.1 Problem Analysis

We ran a full-data EDA over all six source files (about 24.2M records) and the training ground truth. These findings shaped the design:

- **Scale.** Train has 2,206,821 S1 records and 10,320,219 S2+S3 records; test has 1,732,544 S1 and 9,969,589 S2+S3. Brute-force comparison (about 1.7×10¹³ pairs on test) is impossible, so blocking is mandatory.
- **Match structure.** Each S1 entity has 3.46 matches on average (median 3), and only **5.6% are singletons**. About 1.7 matches come from S2 and 1.8 from S3. Recall across several matches matters more than singleton detection.
- **Exclusivity.** 7,638,365 positive pairs cover 7,638,365 distinct S2/S3 ids, so **every S2/S3 record belongs to at most one S1 entity**. We enforce this one-to-one constraint at prediction time.
- **Country.** No training match crosses country labels (0 of 7.6M), so blocking stays safely within each country.
- **S1 is clean; all noise is in S2/S3.** S1 has no empty addresses, Indic script or accents. S2/S3 have 3–4% empty addresses, about 2–3% "null"/"<NULL>"/"N/A" placeholders, about 6–7% injected accents (even in US records), 3–4% domains or social handles used as names, DBA/F/K/A aliases, and legal forms moved, bracketed or duplicated.
- **Indic scripts.** About 23% of India S2 names and 13% of India S3 names are in Devanagari, Kannada, Telugu, Tamil, Gujarati or Bengali script, usually transliterations of English words ("राज हेल्थकेयर प्राइवेट लिमिटेड" = "Raj Healthcare Private Limited").
- **Postal codes are almost never present** (about 11% of US addresses carry a 5-digit token, 0.02% of India addresses a 6-digit PIN). House and plot numbers (present in 90–100% of addresses) plus locality words are the address anchors instead.
- **The test set is harder than train.** It has 5.75 S2/S3 records per S1 against 4.68 in train, so about 40% of the test pool are distractors (26% in train). An audit of our test predictions showed that many are **sibling businesses**: the same street, a neighbouring house number (8650 vs 8671), and a name differing only in its legal form ("Cure Grill PC" vs "Cure Grill Inc") or a filler word ("… Partners"). France (15% of test S1) has no training data at all.
- **Real matches are also noisy in the same fields.** House numbers are dropped, zero-padded, altered or extended (e.g. an added "PLOT 642" on every copy); filler words replace name words ("Rocky Center" is a true match of "Rocky Electric"); some names are entirely invented ("DREXAVI") and only the address links them. So no single field can be used as a hard rule.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier, with a metric-aware set decision (pipeline).  
**Core Innovation:** Training-time synthesis of test-like "sibling" distractors, plus sibling-aware features (house-number distance, one-word name difference, name ambiguity). This lets a standard gradient-boosted matcher learn to reject look-alike businesses that the raw training data rarely shows.

Pipeline:

1. **Normalise** every record into canonical name and address views.
2. **Learn aliases** from ground-truth pairs (romanised Indic word → English word).
3. **Block**: generate about 31 candidates per S1 using a multi-key inverted index, a name-only retrieval channel and a reverse search, per country.
4. **Score** each candidate pair with LightGBM over 61 features; average the probabilities of two models trained on different samples.
5. **Decide**: a house-number conflict rule, an exclusivity-aware joint probability, one-to-one assignment and a probability threshold tuned for macro F0.5.

All stages are vectorised with polars (Rust), and string similarities use rapidfuzz (C++), so the full test set runs in about 50 minutes on one 12-core Colab machine.

---

## 3. Candidate Generation (Blocking)

**Normalisation (`src/normalize.py`).**
- **Names:** Indic scripts are romanised with a Unicode mapping that covers all nine Brahmic scripts (they share Devanagari's code-point layout), with schwa deletion so "लिमिटेड" becomes "limited". Accents are stripped, and junk prefixes, trailing ids, "M/s", handles and domains are removed. DBA/F/K/A aliases are split off, and legal forms anywhere in the name are canonicalised (Pvt/Private/prvhte → pvt, etc.) and then separated from the core name. OCR-style digits inside words are fixed ("C1ark" → "clark").
- **Addresses:** placeholder removal, street-type canonicalisation, state names and codes (including native script), house-number extraction with leading zeros stripped, and ordinal words converted to numbers.
- **Aliases (`src/aliases.py`):** 472 romanised-word → English-word mappings (e.g. "helthkeyar" → "healthcare") are learned by aligning tokens of Indic-script ground-truth pairs. They are applied identically to train and test.

**Blocking keys used (`src/blocking.py`).** Every record emits hashed keys of nine types. An S1 record and an S2/S3 record become candidates when they share a key within the same country:

| Key | Content | Catches |
|---|---|---|
| N | single name token | most names |
| P | pair of name tokens | reordered names, empty addresses |
| T | triple of name tokens | names built from common words, no address |
| A | name token × address word | generic names + locality |
| U | house number × address word | invented or Indic names at the same address |
| B | house number × name token | same number + name, reworded address |
| W | pair of address words | addresses without numbers |
| M | pair of house numbers | multi-number Indian addresses |
| C | concatenated name | handles and domains ("@alikadavis", "navaportillo.com") |

**Scoring and selection.**
- Each shared key adds `type_weight / block_size`, so rare shared evidence dominates. Keys whose block exceeds a per-type cap (200–500 pool records) are ignored.
- For each S1 and each source (S2, S3), we keep the best 80 pairs by this score and compute name and address token-set similarity for them.
- The final set is the **union** of the best 8 by blocking score, the best 8 by name+address similarity, and the best 5 by name similarity alone. The name-only slot recovers address-less matches that would otherwise be crowded out by other businesses at the S1's address.
- A second **name-only retrieval channel** adds the 30 pool records with the highest score from name keys alone (N, P, T, C) before re-ranking, and the caps of the C and T keys are raised to 1,000. This is preset `v16` in `src/blocking.py`.
- **Reverse search (preset `v17`, final):** every pool record also proposes its 2 best S1 records by blocking score over *all* S1 records of the country. A proposed pair not already selected is added when its name+address similarity is at least 1.0 (of 2) or its name similarity at least 0.85. This recovers true copies that were crowded out of their own entity's list by look-alike records; it added 2,534,326 test candidates (+4.9%) and 12,452 final matches, and raised the leaderboard score by 0.0012.

**Candidate pairs generated (test).** 53,832,938 pairs, **31.1 per S1 entity** (51,298,612 before the reverse search), with only 5 of 1,732,544 S1 entities left without a candidate. Against the within-country comparison space (6.72×10¹² pairs), that is a **reduction ratio of 0.999992**.

**How we ensured true matches were not lost.**
- `src/block_eval.py` measures on training data, against the *full* training pool, how many true pairs share any key (coverage) and how many survive selection (recall). It compares configurations and prints examples of missed pairs.
- The final configuration reaches **key coverage 0.995** and **recall 0.972** at 30 candidates per S1. 99.2% of true pairs share a key inside the caps; only 0.5% share no key at all (mostly address-less records with a misspelled name made of common words).
- This process uncovered a real bug. polars 1.35 mis-read list columns of sliced frames, which had silently built keys from the wrong rows and capped recall at 0.48. We restructured the code to never slice list columns, and pinned the library version.

---

## 4. Matching Model

**Features used (61, `src/features.py`), all country-agnostic (no country one-hot), so they transfer to France:**

- **Name features:**
  - Jaro-Winkler, token-set, token-sort and partial ratios on the core name
  - ratios on the concatenated name (for handles and domains) and on a phonetic consonant skeleton
  - DBA-alias similarity
  - token overlap, Jaccard and containment in both directions
  - IDF-weighted shared and missing token rarity (IDF per country, computed on each split's own records)
  - legal-form agreement; record flags (handle, domain, DBA, Indic script)
  - number of name tokens present on only one side, and a "one-word substitution" flag
- **Address features:**
  - token-set and plain ratio on the cleaned address
  - token Jaccard and containment
  - house-number overlap, Jaccard and first-number match
  - **house-number conflict**, and the **log distance between house numbers** (with a "neighbouring number, 1–30 apart" flag)
  - state overlap; empty-address flags
- **Group / context features** (computed over the candidate list of the same S1):
  - blocking score, key-type count, rank within source, gaps to the best candidate
  - duplicate support: other candidates sharing the house number or concatenated name
  - how many candidates carry the S1's own house number, and whether this candidate is in the minority
- **Name ambiguity:** how many S1 records and pool records share the exact name (log scale), and an address-less-and-ambiguous flag.

**Training data.**
- Model A uses 300,000 training and 100,000 validation S1 entities; model B uses 700,000 training and 100,000 validation S1 entities. The split is by entity, and candidates come from the **full** training pool, so negatives are as hard as on test.
- **Synthetic sibling negatives (`src/siblings.py`):** for 90% of sampled entities, 1–3 of their true matched S2/S3 records are copied. The S1 house number is shifted by 1–30 (small shifts most common), and the name gets a different legal form (35%), an added filler word (25%), or a replaced or added word (40%). These records (444,424 for model A, 888,981 for model B) are injected into the training pool as non-matches, so they also appear in validation. At 90% the training pool has about 2.3 non-matching records per S1, close to the test pool's density; raising injection from 40% to 90% was our largest late leaderboard gain.
- The result is 8.86M training pairs for model A and 20.7M for model B (11.4% positive), with about 2.95M validation pairs each.

**Model type:** two LightGBM binary classifiers (MIT licence): 127 leaves, learning rate 0.05, feature and bagging fraction 0.8, early stopping on validation log-loss (best iterations 1,973 and 3,000). Both score the same test candidate set, and their probabilities are **averaged** (`src/predict.py --blend`).

**Threshold selection method.** The decision rule is tuned on the validation entities to maximise **macro F0.5 per S1 entity**, the official metric (`src/decision.py`). We compared plain probability thresholds with an expected-F0.5 subset rule; a threshold of **p ≥ 0.7** was best. Before thresholding, **one-to-one assignment** keeps each S2/S3 record only for the S1 where its probability is highest. A final **house-number conflict rule** sets p = 0 when both records carry house numbers, share none, and p < 0.90. Its cost was measured on validation (−0.0006) and its benefit on the leaderboard (`src/tune_post.py`).

**Exclusivity-aware joint probability (`src/assign.py`, final).** The classifier scores every (S1, record) pair on its own, so a record that fits two S1 entities equally well gets a high probability for both, although at most one can own it. Treating the pair probabilities as independent evidence and allowing at most one owner, the probability that S1 *i* owns the record is q_i = o_i / (1 + Σ_j o_j), with odds o = p / (1 − p). With a single claimant q = p, so nothing else changes; two near-certain claimants (0.99 / 0.98) give 0.66 / 0.33, and neither is predicted, which is right for a precision-weighted metric. Validation only shows this effect when all scored training entities compete for records (+0.0001); on test it drops 0.2% of the predicted pairs, three times more often in France (generic association names such as "Lille Ecole SARL"), and it raised the leaderboard score by 0.0003.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation (100,000 held-out S1 entities, including synthetic siblings for 90% of them):** model A **0.9773** (US 0.9832, India 0.9684; singletons 0.9707, entities with matches 0.9777); model B **0.9782** (US 0.9831, India 0.9709; singletons 0.9716, entities with matches 0.9786). This validation holds about twice as many synthetic siblings as the one used for submissions 4–9, so it is harder.
- **Public leaderboard progression:**

| # | Change | Validation | Leaderboard |
|---|---|---|---|
| 1 | Baseline (blocking affected by the polars slicing bug) | 0.684 | 0.7186 |
| 2 | Fixed blocking; 1/block-size key weights; union re-rank | 0.975 | 0.9256 |
| 3 | + house-number conflict rule (p < 0.95) | 0.972 | 0.9473 |
| 4 | Retrained with synthetic sibling negatives + number-distance features | 0.9766 | 0.9536 |
| 5 | 4 + conflict rule (p < 0.90) | 0.9761 | 0.9573 |
| 6 | Realistic siblings, triple-name key, name-only slot, ambiguity features | 0.9776 | 0.9575 |
| 7 | 6 + conflict rule (p < 0.90) | 0.9770 | 0.9588 |
| 8 | 7 with threshold 0.8 | — | ≤ 0.9588 |
| 9 | 7 + stage-2 group-consistency re-scoring (`src/stage2.py`) | 0.9783 (CV) | ≤ 0.9588 |
| 10 | Model A: blocking v16 (name-only retrieval) + siblings for 90% of entities, conflict p < 0.90 | 0.9773 | **0.9606** |
| 11 | 10 with conflict rule p < 0.95 | — | ≤ 0.9606 |
| 12 | Model B: as 10 with 700k training entities | 0.9782 | 0.960 |
| 13 | Average of models A and B, conflict p < 0.90 | — | 0.9608 |
| 14–15 | Probes: France / India predicted empty (per-country diagnosis) | — | 0.829 / 0.579 |
| 16 | 13 + exclusivity-aware joint probability | 0.9769 | 0.9611 |
| 17 | **16 with the reverse search in both models (final)** | — | **0.9623** |

Only changes aimed at the test set's distractors and candidate recall (blocking fix, conflict rule, heavier sibling training, joint probability, reverse search) moved the leaderboard. Changes that improved validation without that aim (stage-2 re-scoring, more training data alone, a stricter threshold) did not transfer, because validation still has fewer distractors per entity than test.

- **Where validation loses F0.5** (`src/error_analysis.py`, model 4): 73% of the loss is missed matches and 27% false merges. Two thirds of the missed pairs never reached the candidate set, and the model rejected the rest. India loses about twice as much as the US on missed matches.
- **Common false positives (wrong merges):**
  - **Sibling businesses:** same street, neighbouring house number, name differing only in legal form or a filler word ("Green Rapid King Inc, 39 Fairway Rd" vs "Green Rapid King LLC, 44 Fairway Rd").
  - **Address-less records with an identical but ambiguous name** ("King Starry", "20/20 Optical"), where nothing distinguishes two businesses of the same name.
- **Common false negatives (missed matches):**
  - Candidates with an **empty address and a generic name** built from common words ("Urgent Care Physicians Inc Enterprises", "Gulf State LLC").
  - **Heavily scrambled or invented names** ("Dmaigesostcis", "DREXAVI") with a partial address.
  - **Indic-script names** whose romanisation differs strongly from the English spelling.
  - Real matches whose house number was altered, which the model now treats cautiously because of the sibling training.

---

## 6. Conclusion

A classic blocking-plus-gradient-boosting pipeline reached a leaderboard F0.5 of 0.962 on a noisy, multilingual, 20M-record problem. The largest gains came from measuring before tuning: blocking recall evaluation exposed a library bug that capped the score at 0.72, and an audit of test predictions exposed sibling-business distractors absent from training, which we then synthesised from training data only. Our main lesson is that validation must reproduce the test set's distractor distribution; until it did, our validation score overestimated the leaderboard by 0.05.

---

## Appendix

### A. Code Artefacts

Code ships under `code/business_entity_resolution/`. All paths are configurable through the environment variables `BER_DATA_DIR`, `BER_WORK_DIR` and `BER_OUTPUT_DIR`.

```
src/
  config.py          paths, seeds, thread count
  io_utils.py        tab-separated readers (ids kept as strings) and submission writers
  eda.py             exploratory data analysis report
  normalize.py       canonical name/address views, Indic romanisation (Unicode mapping)
  aliases.py         romanised-token -> English-token aliases learned from training pairs
  blocking.py        multi-key inverted index, 1/block-size scoring, union selection, presets
  block_eval.py      blocking coverage/recall comparison on training data
  siblings.py        synthetic sibling-business hard negatives (training only)
  features.py        61 pair features (rapidfuzz, polars)
  decision.py        one-to-one assignment, threshold / expected-F0.5 rules, official metric
  postprocess.py     house-number conflict rules
  assign.py          exclusivity-aware joint probability (final) and owner-reassignment variants
  stage2.py          group-consistency re-scoring (tried; not used in the final submission)
  train.py           aliases -> siblings -> blocking -> features -> LightGBM -> decision tuning
  predict.py         test blocking -> features -> scoring -> (--blend) averaging -> decision -> output files + validator
  eval_joint.py, eval_blend.py              validation of the joint rule and of the blend settings
  tune_post.py, audit.py, audit_country.py, error_analysis.py, diag_*.py   analysis tools
  patch_names.py     honorific stripping (tried; lowered validation, not used)
tests/               normalisation regression tests; synthetic data generator for smoke tests
tools/validate_submission.py   organisers' validator (unchanged copy)
```

**Reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`:**

```bash
pip install -r requirements.txt
export BER_DATA_DIR=/path/to/dataset BER_WORK_DIR=work
python -m src.normalize                                                                        # ~5 min
python -m src.train --blocking v16 --siblings 0.9 --model-dir work/model_v3                    # model A, ~25 min
python -m src.train --blocking v16 --siblings 0.9 --n-train 700000 --model-dir work/model_v4   # model B, ~60 min
# copies of both model folders that use blocking v17 (v16 + reverse search) on test
for m in model_v3 model_v4; do
  mkdir -p work/${m}r && cp work/$m/model.txt work/$m/aliases.json work/${m}r/
  python -c "import json; p=json.load(open('work/$m/params.json')); p['blocking']='v17'; json.dump(p, open('work/${m}r/params.json','w'))"
done
python -m src.predict --model-dir work/model_v3r --post conflict_p90 --out-dir work/out_a      # blocking + scoring, ~60 min
cp work/model_v3r/test_cands.parquet work/model_v4r/
python -m src.predict --model-dir work/model_v4r --post conflict_p90 --out-dir work/out_b      # scoring, ~48 min
python -m src.predict --model-dir work/model_v3r --blend work/model_v4r --post conflict_p90 \
       --assign joint --out-dir output --validator tools/validate_submission.py               # final, ~3 min
```

Timings are on a Google Colab L4 High-RAM runtime (12 vCPU, 53 GB RAM); no GPU is required. Seeds are fixed (`SEED = 42`).

**Compliance.**
- Only the provided training and test files are used. There are no external databases, APIs, geocoding, gazetteers or internet lookups at any stage, and after installing dependencies the pipeline runs fully offline.
- All normalisation vocabularies (legal forms, street types, state names) are short hand-written lists in the source code. Aliases and synthetic siblings are derived from the training data only.
- No pretrained models are used. Libraries: LightGBM (MIT), polars (MIT), rapidfuzz (MIT), NumPy (BSD), psutil (BSD).

### B. Additional Results

- **Blocking configurations** (20,000 training S1 records against the full training pool):

| Configuration | Recall | Candidates per S1 |
|---|---|---|
| v6: union 8+8 of top 50 | 0.9646 | 24.8 |
| v8: v6 + triple name key | 0.9652 | 24.7 |
| v9: + name-only slot 8+8+4, of top 50 | 0.9679 | 26.8 |
| v10: 8+8+4, of top 80 | 0.9698 | 27.8 |
| v14: v10 + name retrieval 20 | 0.9704 | 28.6 |
| v15: 7+7+5, name retrieval 20 | 0.9701 | 26.7 |
| **v16: 8+8+5, name retrieval 30, C/T caps 1,000 (final)** | **0.9719** | **30.2** |
| v11: 6+6+4, of top 80 | 0.9670 | 21.7 |
| v12: 8+8+8, of top 80 | 0.9717 | 32.5 |
| v13: 5+5+3, of top 80 | 0.9630 | 17.5 |

- **Post-processing trade-off (model of submission 7, validation):**

| Rule | Validation F0.5 | True pairs removed | False pairs removed | Test pairs removed |
|---|---|---|---|---|
| none | 0.97763 | 0 | 0 | 0 |
| conflict (p < 0.90) | 0.97702 | 917 | 171 | 32,098 |
| conflict (p < 0.95) | 0.97630 | 1,656 | 216 | 48,366 |
| any conflict | 0.95871 | 14,136 | 267 | 210,647 |

- **Most important features (LightGBM gain):** blocking score, key-type count, house-number Jaccard, house-number log distance, name Jaro-Winkler, neighbouring-number flag, address containment, IDF of the rarest missing token, concatenated-name ratio.
