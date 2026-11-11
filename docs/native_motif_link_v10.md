# PhonoWeave v10 — automatic cross-recording acoustic motif association

**Research-only.** The code finds recurrent **acoustic regions**, not verified phones, IPA symbols, pronunciation inventories, or synthesizable units. No manual boundary annotation or external dictionary is required to run it. Its current real-data demonstration uses v9's historically **OTO-curated 600-ms audio snippets**; it must not be described as full-recording, independent-phone recognition.

## New algorithm

1. **Input:** 48D log-mel region embeddings saved by v9 (`raw_spectral_embeddings.npz`, `motif_regions.json`) or a new ZIP containing unaltered mono PCM16 full WAV files. In the second path, v9's waveform-only activity/change-point feature functions are run over every eligible WAV; no OTO timing, alias or transcript is read.
2. **Source-disjoint split:** conservatively group same-filename original recording families across banks/styles (or use v9's archival clip-family IDs). No family crosses training (60%), validation (20%), or audit (remaining 20%). An identical PCM hash attributed to different recording families is refused as potential leakage. **Banks are not distinct speakers.**
3. **Learn acoustic geometry:** shrinkage-scaled 48D features using *training-only* statistics. Evaluate k-means orders **k=1..7**, with seeded restarts. All models and associated normalization use training families only; validation controls the number of motifs.
4. **Choose k or abstain:** screen nontrivial models for group support, minimum region counts, heldout validation reconstruction gain, and separation. Apply a model-complexity penalty, then require at least 0.75 median adjusted Rand index under deterministic training-family subsampling. If no multi-motif model survives, use k=1, **not a claim that only one phone exists**. These heuristic thresholds are not perceptual or universal linguistic cutoffs.
5. **Unknown class:** estimate per-centroid novelty radii on training + validation (not audit); mark far-away or ambiguous regions as `unknown_or_ambiguous_acoustic_region`, never force a user label. Report audit coverage, not just accuracy-like geometry scores.
6. **Cross-recording links:** use three seeded random-projection approximate retrieval indices to find candidate neighbors efficiently, then calculate exact normalized spectral distances. Link only mutually retrieved nearest neighbors belonging to the same unverified motif, from different PCM hashes and different recording families, within radii determined without audit data. Approximate candidate retrieval can **miss** some valid geometric neighbors; links do **not** prove phone identity.
7. **Explicit confound audit:** measure normalized mutual information between motif IDs and acoustic-region order, bank and source family. Repeat exploratory model selection within region-order strata after the global model is frozen. Such post-hoc probes are not independent confirmation and do not alter global model selection.

## Usage

On frozen v9 experiment products:

```bash
python -m phonoweave.native_motif_link_v10 \
    --v9-results /path/to/v9_direct_wav_results \
    --output /path/to/new_v10_results
```

On unanchored complete WAVs when the original local voicebank archive is available:

```bash
python -m phonoweave.native_motif_link_v10 \
    --full-wav-zip /path/to/private_original_full_wavs.zip \
    --output /path/to/another_new_v10_results
```

The full-WAV path is implemented and covered by synthetic WAV tests, **but has not been experimentally run on the unavailable full 175-recording LynneQing audio archive**. ZIP must contain normal PCM16 mono WAVs. The filename-based family grouping deliberately merges identical WAV basenames across subbanks for split safety; it may be conservative for independent recordings that share filenames. Do not commit private audio or original voicebanks to GitHub.

Outputs: `summary.json`, `regions.json`, `cross_recording_links.csv`, `frozen_centers.npz`. They contain source paths and hashes and thus should be reviewed before publication. They do **not** contain raw WAV audio, original OTO files, or new user alias assignments. An existing output directory is never overwritten.

## Actual retrospective v9 data continuation (2026-11-10)

- 503 acoustic regions, from 168 historical OTO-selected 600-ms PCM clips in seven LynneQing subbanks. Source collection has **23 recording clip families**, from one historical speaker source; this is not seven-speaker validation.
- Train / validation / audit: **315 / 104 / 84 regions** across **14 / 5 / 4 families**. These audit groups are not a fresh prospective test corpus: v9 had already examined all of the same clips before v10 was developed, and v10 thresholds were implemented in response to exploratory outcomes. Treat the audit only as an internal source-disjoint diagnostic.
- Auto-selected **3 acoustic patterns** from k=1..7, with group-resample median validation ARI **0.97078**. Validation reconstruction gain for k=3 vs k=1: **0.39593**. This is a geometric improvement, not phoneme recognition accuracy.
- Of 84 retrospective audit intervals: **70 candidate assigned / 14 unknown**, audit assignment coverage **83.333%**. **328 reciprocal cross-family acoustic links** involve **341 of 435 non-unknown regions** over the entire dataset. These links are not correct-phone scores.
- **Critical confound:** motif vs `region_index` NMI **0.74658**, much higher than motif vs recording family (**0.03475**) or bank (**0.03768**). One class is overwhelmingly the central audio stage; the others largely separate the outer two positions. This pattern is consistent with *within-clip temporal phases*, NOT evidence of three phonemes.
- After splitting post-hoc by acoustic order, index 0 (**168 regions**) and index 1 (**167 regions**) yielded no reliable >1 subpattern. Index 2 (**168 regions**) yielded a provisional three-subpattern decomposition, group-resample median ARI **0.83783**, subject to context/provenance confounding. Do not infer three user-native pronunciations.

## Acceptance restrictions and next bottleneck

- No autogenerated alias, phone symbol, pronunciation dictionary, waveform edit or split/merge decision is authorized.
- Do not confuse high ARI with semantic/phonetic correctness; not even source-family disjointness ensures different speakers or new recording conditions.
- The 175 original full WAVs, a separately sourced recording script and actual multi-speaker recordings are still absent from this environment. v10's next evidence gate should prioritize running the full-audio extraction on those original WAVs and assessing **recurrent motifs across different carrier contexts** before deciding whether any within-position subtype is stable.
- Research hyperparameters are post-hoc exploratory choices. Freeze them and recruit *new independent recordings* before confirmatory publication claims.

## Synthetic tests

Run `pytest -q tests/test_native_motif_link_v10.py`. Tests cover automatic 3-pattern recovery, 1-pattern abstention, whole-family split, blind audit invariance, scaling/threshold isolation, name/label nondependence, mutual cross-source linking, unknown outliers, pathological input, ZIP security, artifact identity, and refusal to overwrite. A green result demonstrates software behavior only—not phone segmentation quality.