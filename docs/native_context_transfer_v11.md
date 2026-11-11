# PhonoWeave v11 — cross-carrier acoustic recurrence and style transfer safeguards

**Status: exploratory unsupervised acoustic analysis, not phone identification.** No known phoneme truth, no manual phone boundaries, no guaranteed language-independent phone segmentation, no authorized alias or recording-list changes. These recordings originate from a previously OTO-curated snippet collection even though the feature extractor uses audio alone.

## Purpose

v10's position/order audit found that the global `k=3` clusters were primarily temporal stages within clipped audio (order-vs-motif NMI ≈0.747). An exploratory post-hoc attempt at segment-order index 2 found three subclusters with source-group subsample validation ARI ≈0.838. **That is not enough**: recording style can be a confound and original carrier contexts must be held out by family. v11 adds complementary, conservative tests; it does not claim to correct those features into phonetic identities.

## New algorithm

1. Load frozen 48D log-mel segment embeddings from v9, or directly extract unanchored segments from a full PCM16 WAV ZIP via the v10 extractor. Optional v10 provenance check requires an exact region-by-region match on WAV entry, PCM SHA-256, ordinal and endpoints plus the same original ZIP SHA-256. No `oto.ini` or IPA is consulted.
2. Stratify by acoustic region order **only** (e.g., first/middle/last segment in an automatically extracted 600-ms clip), not linguistic phone position. Split whole `clip_group`/carrier families 60/20/20 into train/model-selection validation/retrospective audit, preserving copies from all seven subbanks in the same fold. Grouping metadata is only used for split and audit, not as acoustic features. The seven banks do **not** denote seven independent speakers.
3. Independently select `k=1..4` acoustic modes within each stratum by v10's development-only rules (K-means, source-group subsamples, novelty checks). With `k>1`, fit separate `lu` and `mang` style prototypes on **training** families; compare their partitions of **validation** families using adjusted Rand index (ARI). Require ARI ≥0.65 before calling the split even *exploratorily cross-style reproducible*. Threshold is a fixed, exploratory guard selected after v10 observations; it is **not** a validated phonetic threshold. ARI on audit families is descriptive and cannot rescue validation failures.
4. For audit regions, find neighbors of the same *unverified* v10-style acoustic mode in the **development** carrier families only, excluding the same family and exact same PCM source. Calibrate a squared-distance radius on development-only nearest neighbors from at least two distinct carrier families. An acoustic echo additionally needs matches in **two styles** and **two different carrier families**. This is recurrence evidence only, **not the same-phone precision**.
5. Test whether stage-0 audio embeddings predict stages 1/2. Use per-bank train-only means as the baseline and linear ridge on 48D acoustic neighbor vectors. Select regularization from (10, 100, 1000) using validation only; compare to 24 negative controls that permute training neighbors **within each bank**, each allowed its own validation-best regularization. Report heldout improvement descriptively, but cannot claim causation or phonetic content. This analysis is exploratory and the resulting p-value is coarse.
6. Abstain on insufficient coverage, single-style/single-mode inference, inconsistent source hashes, missing independent carrier families, no replicable subdivisions, or no clear contextual effect. Source audio and labels are never edited.

## Reproduce using stored historical features

```bash
python -m phonoweave.native_context_transfer_v11 \
  --v9-results /path/to/v9_audio_results \
  --v10-results /path/to/v10_audio_results \
  --output /path/to/NEW_v11_results
```

To run completely unanchored extraction when an independently held full-WAV archive is available:

```bash
python -m phonoweave.native_context_transfer_v11 \
  --full-wav-zip /path/to/private_original_recordings.zip \
  --output /path/to/NEW_v11_full_recording_results
```

Full-WAV functionality is **synthetically tested**. It has not yet been applied to the missing complete LynneQing original voicebank WAV archives. Both input paths refuse pre-existing output directories. The v10 ZIP member-order sorting bug discovered in v11 is fixed in the bundled v10 dependency, with a multi-WAV regression test. No audio needs to be uploaded to GitHub.

## Historical LynneQing data results (2026-11-10)

- 168 OTO-selected 600-ms original-source excerpts from one historic singer (7 subbanks; 2 style families: `lu`, `mang`), with **503** independently proposed unanchored spectral segments. 23 named carrier templates (e.g., `a_to_sh`, `ai_to_sh`, `i_to_sh`), which are **historical filename metadata** and are not presumed true phone boundaries. Recorded as 23 clip families and held out by family across bank/style copies.
- Stage 0: 168 regions. v10 order-specific selection `k=1`, with **27/28** audit intervals known under the broad acoustic-mode novelty rule, and **14/28** further meeting multi-carrier-and-style echo criteria. This does not establish a phoneme class.
- Stage 1: 167 regions. `k=1`, **25/28** known, **17/28** meet multi-carrier-and-style echo criteria. Acoustic prediction from stage 0: **+0.00108** validation improvement vs per-bank baseline (no reliable gain); descriptive audit **+0.10073**. No subtype authorized.
- Stage 2: 168 regions. Exploratory `k=3` with source-group resample validation ARI **0.83783**. **Independent model training in the two voice styles** yields validation ARI **0.11065**, descriptive audit ARI **0.03037**. Hence **v11 explicitly rejects the subtype interpretation**. Of the 28 heldout stage-2 intervals, 24 were assigned as known under v10's provisional novelty radii, only **6/28** meet the two-carrier/two-style recurrence condition. Acoustic stage-0 context prediction yields validation improvement **−0.05101** vs bank baseline (not reliable); descriptive audit **−0.26670**.
- The models cannot infer whether stage 1 or 2 equals an acoustically exact `sh` segment. An originally curated filename `*_to_sh` names a selection criterion, not an independently observed boundary. **No confirmed distinct phonetic realization, synthetic-unit split/merge, or recording scheme revision.**

**Selection caveat:** Because earlier versions looked at these same source excerpts, and this cross-style guard was devised in response to v10 failure, the entire v11 analysis is retrospective, including the source-family audit split. Never present these outputs as confirmatory generalization or real-phone recognition accuracy. A prospective independent multi-speaker dataset with uncurated full WAVs and blind boundaries is needed for such claims.

Outputs are a JSON `summary.json` and `audit_region_recurrence.json` (metadata, relative audio entry names, hashes, and provisional recurrence statuses; no raw audio). These files may reveal original local paths and should be reviewed before external publication.

## Tests

```bash
pytest -q tests/test_native_context_transfer_v11.py
pytest -q tests
```

The v11 tests cover source and PCM identity, cross-style success and failure, carrier support, audit-only radii, ridge synthetic signal/no-signal, within-bank shuffled controls, full-WAV ZIP input, duplicate/invalid source identity, source exclusivity, and abstention. Green synthetic tests prove implementation invariants, not empirical phone accuracy.