# Synthesis Relevance — acoustic separability v0.5

## Scope

Objective, no-listening investigation of whether Mandarin `sh` (optionally `s`, `x`) differs acoustically in plain versus rounded OTO contexts. All input is existing voicebank WAV + OTO; no source files or production voicebank decisions are modified, and no audio is generated.

This **does not yet establish the necessity of separate UTAU synthesis units**. It tests a weaker, falsifiable property: context-conditioned acoustic separability in the recorded voicebanks.

## Acoustic observations

Parse aliases using existing Mandarin and prefix-map modules. Candidate consonant interval is `[offset, offset + preutterance]`. Reject intervals shorter than 110 ms; remove 12 ms at the beginning and 22 ms at the end and compute 25 ms Hann spectra at three equally spaced interior positions. Each position yields spectral center of gravity (kHz, 1.5–11 kHz), high-band (>=5 kHz) power fraction, and spectral flatness. These nine objective features capture both spectral shape and coarse time trajectory, without using audio playback.

These are **OTO-based approximate** consonant regions. They are not independent acoustically verified frication-only boundaries; the estimated interval may include neighboring vowels or silence. This limitation remains central when interpreting any positive result.

## Validation and controls

- Deduplicate overlapping OTO aliases by source WAV, approximate alignment coordinate and context.
- Use 5-fold **physical-WAV-grouped** cross-validation: no segment from the same source WAV appears in both train and held-out test. Within each training fold only, standardize features and fit a nearest-centroid plain/rounded classifier. Report both class recalls and balanced accuracy.
- The folds are approximately stratified by the source WAV's available context composition. If a fold cannot support both contexts, report insufficient evidence rather than fabricating a score.
- Independently, compare the mean rounded-minus-plain feature difference **within the same original WAV** and pitch subbank, averaging multiple OTO aliases inside each WAV/context. Apply joint WAV-level sign-flip permutations (2,000) to the paired nine-dimensional difference; report a multivariate statistic and permutation p-value, plus descriptive feature differences.
- Report each pitch subbank independently and retain per-voicebank estimates. A physical WAV is one paired sampling unit, not as many statistically independent units as OTO aliases.

## Interpretation

A grouped classifier above 50% and a small within-WAV permutation p-value may be consistent with context-dependent consonant realization. However, differences in following vowel, OTO timing, phonation and subbank conditions remain possible explanations, especially if OTO intervals include adjacent sound. A non-significant result does not demonstrate interchangeable recording units; a significant result does not demonstrate that a split improves downstream synthesis. Neither effect is a license for automatic `synthesis_inventory.yaml` split/merge decisions.

Next steps, contingent on stable and reproducible results, are segment-boundary sensitivity tests, negative controls and an objective full-synthesis merge-cost study. Additional independent speakers and engine-specific controls would be needed for strong generalization claims.

## Run

```bash
python -m pytest -q
python -m phonoweave.synthesis_relevance_acoustic_separability_cli \
  ~/Downloads/phonoweave_voicebanks.json \
  --base sh \
  --output ~/Downloads/phonoweave_sh_separability_v0.5.json
```
