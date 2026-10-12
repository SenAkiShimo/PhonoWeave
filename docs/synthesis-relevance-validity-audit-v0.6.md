# Synthesis Relevance v0.6 — OTO validity audit and conservative negative controls

## Motivation

The v0.5 objective `sh` separability pilot showed WAV-grouped binary balanced accuracy between 0.638–0.923 for six evaluable voicebanks, and within-WAV paired sign-flip p≈0.0005 in two voicebanks. This only supports preliminary OTO-aligned spectral association; the extraction rejects most Cha Yue observations. Existing v0.5 diagnostics lump several rejection causes.

## Frozen inputs and failure audit

Keep original v0.5 `_features` implementation, 110 ms minimum interval, 12 ms onset exclusion, 22 ms right exclusion, 25 ms Hann spectra and identical deduplication key. Record exact rejection classes, OTO duration bins, subbank and context failure patterns. Replay `extract_bank` and demand identical Observation vectors before reporting; diagnostic code is not allowed to silently increase coverage.

## Robustness checks

- WAV-grouped label null: reproduce v0.5's exact five held-out WAV-group folds, flip all plain/rounded class labels within each physical WAV together, then restandardize training folds, reestimate two class centroids and recompute held-out balanced accuracy. Run 399 valid permutations and explicitly count rejected trials due to lacking either class in a fold. The permutation distribution conditions on v0.5 fixed folds and feasibility; its p-value is a **descriptive conditional null**, not a cross-speaker inference test. Class-flipping within same-WAV pairs means the within-WAV context structure is preserved, but subbank/phonetic-final confounds remain. This particular whole-WAV flip can be conservative in banks where most WAVs contain both class examples.
- Right-edge trim sensitivity: recompute exact nine features with the final OTO edge shortened by 20 or 40 ms, never move the left edge. Compare original versus trimmed classifier only on identical accepted event identities. Report retained fraction and paired balanced-accuracy changes. This is *not* validated consonant-only segmentation and may remove portions of legitimate frication.
- Leave-one-subbank-out: use all other pitch-layer subbanks for training and report balanced accuracy on each excluded subbank, when both classes exist. Do not claim this is independent speaker transfer.
- No signal playback, no audio exports, no original OTO modifications, no automatic split/merge recommendation.

## Known limitations

The OTO `preutterance` position is a renderer-control timing parameter, not a phonetic annotation. Acoustic differences could arise from downstream vowels or recording-list conventions. Physical WAV grouping blocks within-file leakage but does not create independent speakers. The sign-flip test and group-flip null are dependent on exchangeability assumptions and should be interpreted alongside per-bank sample sizes, window exclusion patterns and effect directions. A successful discriminative classifier does not imply the source units cannot be merged for synthesis.

## Run

```bash
cd ~/Downloads/PhonoWeave
source .venv/bin/activate
python -m pytest -q
python -m phonoweave.synthesis_relevance_validity_audit_cli \
  ~/Downloads/phonoweave_voicebanks.json \
  --base sh --permutations 399 \
  --output ~/Downloads/phonoweave_sh_validity_audit_v0.6.json
```
