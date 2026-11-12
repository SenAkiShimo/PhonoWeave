# User-native temporal context validation v3

## Scope and non-goals

A controlled follow-up to `discover-user-units-batch` v2. Its input symbols are **opaque user-authored alias surfaces** (`i`, `ir`, `i0`, `i2`, Unicode tokens etc.), not standardized IPA phones. The test measures how often a simple classifier recovers **existing labels** from source-linked acoustic observations. It is not automatic phone segmentation, new-phoneme discovery, or evidence that a unit should be recorded separately.

## Dataset and original-v2 binding

- LynneQing: 7 pitch subbanks from the **same authored DiffSinger voicebank**, styles Mang (G3, D4, G4, D5) and Lu (C4, F4, B4). These are not 7 independent speakers or independent sessions.
- Full original voicebank has 13,539 OTO rows; the v2 **two-token surface** scanner accepted 7,860 candidate audio windows, 23 left surface labels, 47 right surface contexts, 25 same-named source WAV templates across banks. Two v2 windows were abstained.
- The v3 executable **requires** exact v2 `source_index.csv` ↔ `feature_vectors.json` row provenance, verifies each raw archive SHA-256 and source WAV PCM SHA-256 against the frozen index, and now checks original OTO line number, source WAV, unmodified user alias, left and right labels, and relative anchor. It never changes original WAV, `oto.ini`, or aliases.

## Experiment: avoid test-set-led time-window choice

Old baseline: frozen 70 ms [-125, -55] relative to the OTO anchor `offset + (preutterance + overlap)/2`. None of these times should be called a true phoneme boundary.

Seven prespecified candidate representations were evaluated on **17 development source-WAV-template groups**: original 70 ms; early context; wide mean; wide trajectory; late context; full trajectory; and a local stable-region heuristic. These were built from five 80 ms segments of PCM measured relative to the same OTO anchor: far [-340,-260], early [-270,-190], mid [-200,-120], late [-130,-50], edge [-80,0] ms. Window features are median 12-dimensional, normalized Mel-DCT spectral shape. Longer contexts can contain coarticulations and phonetic material belonging to the next token.

For every fold, (a) the complete named parent WAV template is held out **across all 7 banks**, (b) training prototypes use *only* non-heldout source templates, (c) bank-level mean correction is learned from **training examples only** (if an entire bank is unseen, use the global train mean), (d) labels are used for evaluation and supervised prototypes, not for segment choice, and (e) dev selection is based on pooled macro recall, breaking ties by pooled accuracy. The eight frozen source templates were selected by deterministic hash of the **template filename only**, independent of score; no test labels were used to pick a representation in v3.

The selected representation is `late_context`: the concatenated summaries of [-200,-120], [-130,-50], and [-80,0] ms. The distinct time segments allow vowel-to-consonant spectral trajectories to enter the comparison while avoiding a single 70 ms snapshot.

## Real measured results

| Setting | Frozen v2 70 ms | Selected late-context | Meaning |
| --- | ---: | ---: | --- |
| 17-group development leave-one-template-out | 36.37% | 42.25% | model selection, not pristine heldout |
| **8 frozen heldout templates**, trained on 17 dev groups | **847/2,044 = 41.44%** | **958/2,044 = 46.87%** | single untouched v3 holdout for method selection |
| Full 25-template leave-one-out (post-selection diagnostic) | 37.79% | 42.70% | not an independent confirmatory metric |
| Leave-following-context-out | 39.95% | 45.97% | helps rule out simple memorization of right-context labels |
| Leave-pitch-bank-out | 29.52% | 33.46% | genuinely training-only normalization; still poor transfer |
| Leave-entire-style-out | 28.92% | 32.90% | clear limitation across authored styles |

The development selection primary metric was **macro recall**, baseline 37.88% versus selected 43.93%. The heldout macro recall is 38.18% versus 43.91%.

All 8 source templates in the v3 heldout comparison had a positive count delta. Group-bootstrap, resampling the 8 template names (25,000 seeded draws), gives a **descriptive** 95% range of +3.54 to +8.54 percentage points for the absolute accuracy improvement. It is not a population-level independent-speaker confidence interval; all observations belong to one authored DiffSinger voicebank. The prior v2 **38.0%** source-template result used **unlabeled evaluation-bank centering**, while the new strict full-template figure uses train-only bank centering; do not compare those percentages without naming the changed normalization.

## Evidence gate and limitations

The result justifies a **better acoustic diagnostic representation**, not unit splitting/merging or automatic source annotation. The phonetic content and onset of each OTO-relative window are not independently adjudicated. The late window ends at the OTO anchor and may contain context from the next consonant. Styles, pitches and underlying synthetic performances share generation structure. The 25 source WAV templates and all v2 labels were known from earlier research; the v3 hash partition protects *within this model selection*, not absolute historical test-set freshness. No output authorizes changing production aliases or synthesis inventories.

The program writes `evaluation.json`, `robustness.json`, `holdout_by_source_wav.csv`, checksum-bound `temporal_features.npz` and its SHA-256. No original audio is included or uploaded to GitHub. The CLI and unit tests are part of the existing PhonoWeave Python package.

## Reproduction

```bash
phonoweave discover-user-unit-timecourse \
  --batch-result /path/to/v2/results/lynneqing_seven_subbanks \
  --manifest /path/to/seven_archive_manifest.json \
  --archive-root /path/to/private_zip_archives \
  --output /new/empty/output_directory
```

No MFA, G2P or IPA is needed. Reported test metrics do **not** count as accuracy of true phone boundaries; no new alias is automatically minted.

## Next scientific dependency

A universal user-language segmenter needs a temporally grounded **actual sound-unit occurrence model** trained/evaluated against independent phone/time annotations, while continuing to preserve user-native identifiers. The short pilot has not yet supplied such boundary ground truth; more template-relative classification alone cannot solve automatic segmentation.
