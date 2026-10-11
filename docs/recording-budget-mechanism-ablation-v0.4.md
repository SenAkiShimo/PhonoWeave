# Recording-budget mechanism ablation v0.4

## Question

Does the observed-feature adaptive policy obtain its apparent advantage from *previously acquired acoustic dispersion*, or mainly from candidate metadata such as final novelty?

The v0.2 disjoint-WAV folds (partition seed 20261011), four budget fractions, original acoustic feature scaling and held-out distance remain unchanged. This post-hoc exploratory ablation follows the v0.3 random-null analysis. No coefficient is optimized against the voicebank.

## Conditions

| Variant | Context balancing | Acquired acoustic dispersion | Unobserved-final novelty |
|---|---|---|---|
| full_adaptive | yes | yes | yes |
| no_acoustic_dispersion | yes | no | yes |
| no_final_novelty | yes | yes | no |
| metadata_balancing_only | yes | no | no |

The full condition dispatches to the original `select_sequence` implementation and must reproduce v0.2 exactly. The metadata-only condition must reproduce `fixed_balanced` exactly. An acoustic feature is accessed only after its WAV has been selected. The novelty feature refers to known candidate final metadata, not unseen WAV spectra.

Each variant is evaluated at the same per-fold WAV budget against the same held-out WAVs. Negative `minus_full_adaptive` indicates that removing a term improves acoustic distance. Fold-by-fold results are reported to avoid confounding a global average with a single exceptional fold.

## Interpretation

These ablations isolate source-code contributions, not causal biological phonetic mechanisms. The three-way interaction of balancing, dispersion and novelty means disabling one term may change the entire acquisition sequence. No inference about singing-synthesis quality, independent singers or microphone generalization follows from this single-voicebank evaluation.

## Run

```bash
python -m pytest -q
python -m phonoweave.recording_budget_mechanism_ablation_cli \
  "/path/to/voicebank" \
  --output ~/Downloads/phonoweave_budget_ablation_v0.4.json
```
