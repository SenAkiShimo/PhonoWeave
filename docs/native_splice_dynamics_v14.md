# PhonoWeave v14 — waveform-rendered splice dynamics (research-only)

## Status and scientific question

**Exploratory negative result, not a singing-quality study.** This module extends the unlabeled acoustic-motif line v9–v13 to **actual rendered PCM waveform splices** and separately evaluates how accurately a donor's subsequent short-time spectrotemporal trajectory matches the source recording's withheld continuation. No new phoneme labels, IPA normalization, OTO edits, split/merge decisions, or reclist output is authorized.

Important: the 168 source clips are historical *OTO-curated* 600-ms extracts, not the 175 intact original recordings. A cut at the v9 unanchored third-region start is **not verified as a CV boundary** or any linguistic phone boundary. The same singer's seven pitch/style subbanks are **not seven speakers**. All source groups have been inspected in earlier work; test sets are retrospective, not an untouched independent confirmation.

## Method

- Inputs: frozen v9 `motif_regions.json`, `raw_spectral_embeddings.npz`, `summary.json` and the *real* source ZIP (`PhonoWeave_Actual_Phone_Segmentation_v0.1.zip`). Before any processing, verify the entire source ZIP against the frozen v9 archive SHA-256. Verify every individual decoded PCM SHA-256 against the frozen region and v13 clip identities.
- Eligible source clip: all three unanchored v9 acoustic stages present; each selected stage-2 cut has >=90 ms audio to its left and >=150 ms to its right. One clip lacking the full three-stage sequence is excluded. Results: 167 valid short WAVs at 44.1 kHz.
- Holdout: the already-fixed v10 group split (14 train, 5 validation, 4 retrospective audit **carrier families**), with each family kept intact across all subbanks. Across 28 retrospective audit clips and 34 validation clips, the physical source PCM hashes are distinct from train donors.
- Every query sees **exactly 4 training donor candidates**: the same-bank stage-2 training clip windows closest to the train-only stage-2 spectral centroid, as in the v13 `pooled` baseline. The pooled arm uses its first central donor. The seam-aware arm chooses **one of precisely the same four donors** using only the left-hand source audio; the target's post-cut right audio is never available at selection time. This is a matched *four-candidate* comparison, not an equality of the ultimate recording inventory, and the baseline does not exploit all four candidates equally.
- Seam-aware fixed score: 0.7 × short-time 24D log-mel boundary mismatch normalized by train natural-seam medians + 0.3 × absolute log RMS gap normalized by train medians; add a 0.2 × normalized autocorrelation-periodicity gap only if both sides pass a conservative periodicity gate. Weights, cut lengths and candidate budget are fixed; **no audit or withheld right audio can tune this rule**. The periodicity feature was largely unavailable at these fricative→later-stage seams (0% audit coverage). It is not F0 tracking.
- The left part is 90 ms from the query audio; the donor right is 150 ms starting at its own frozen stage-2 boundary. Render an actual short mono 44.1-kHz PCM WAV with 8-ms equal-power crossfade; no true synthesizer engine, pitch matching, prosody control, vocoder or resampler.
- Evaluate five separate diagnostics: the scorer's own seam mismatch (expected to go down when optimized); rendered localized peak/RMS; rendered maximum sample-step/RMS near splice; absolute log energy deviation against neighboring windows (distance from ratio 1, **not** lower energy ratio); and a truly independent proxy, 8-frame 24D log-mel trajectory MSE of each donor's **post-crossfade** 100-ms audio against the query source's withheld real continuation. Train-only mean/scale standardize the continuation trajectory. Do not merge these into a subjective quality score.
- All scores are paired on the same query and four-candidate pool, group-averaged first within carrier family and then equally across carrier groups. Do not automatically turn validation outcomes into model changes: validation and audit are reported descriptively; the historical audit was seen in earlier studies.

## Reproduce

```bash
python -m phonoweave.native_splice_dynamics_v14 \
  --v9-results /private/frozen_v9/audio_results \
  --audio-zip /private/PhonoWeave_Actual_Phone_Segmentation_v0.1.zip \
  --output /private/NEW_v14_output \
  --render-examples
```

No existing output folder is overwritten. Default: **do not** render audio files. If `--render-examples` is chosen, pairs of illustrative PCM WAVs are placed locally in `illustrative_splices/` (the selected historical `ai_to_sh` group, one pair per subbank). Never commit audio or source-private archive paths to public GitHub. `audit_comparisons.csv` exposes source identities and hashes and should also be treated as locally retained research data.

## Observed retrospective results

| Grouped endpoint | Validation pooled → aware | Retrospective audit pooled → aware | Relative audit gain (positive better) |
| --- | --- | --- | ---: |
| Own seam cost | 1.08148 → 0.90659 | 0.95557 → 0.80031 | +16.25% |
| Rendered peak/RMS | 3.43162 → 3.07090 | 3.41629 → 3.17035 | +7.20% |
| Rendered max sample-step/RMS | 0.81511 → 0.99381 | 0.93551 → 1.17530 | **−25.63%** |
| Absolute log energy deviation | 0.16272 → 0.16298 | 0.15760 → 0.16720 | **−6.09%** |
| Withheld real continuation log-mel trajectory MSE | 0.61667 → 0.69549 | 0.34718 → 0.42507 | **−22.44%** |

One out of four audit carrier groups showed improved continuation MSE; all four had worse max-sample-step/RMS. Seam-aware selected a different donor from pooled in 26/28 audit clips. The seam scorer itself improved in all four groups, but **optimization of its own predictor is not an independent quality measurement**. The remaining metrics are *engineering proxies*, not human-audited transition naturalness. Short-time noise and corpus context confound them. No end-to-end singing synth or listening validation is performed.

## Decision

**Reject automatic inventory/alias changes.** Smoothing a proposed spectral discontinuity alone does not imply natural continuation or rendered transient improvement. This negative result motivates future experiments with proper aligned CV boundaries, source-balanced donor pools and actual synthesis engines, ideally with source-independent recordings and perceptual preference studies. All results remain auditable, hash-grounded, and non-destructive.