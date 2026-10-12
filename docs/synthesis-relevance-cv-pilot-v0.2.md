# Synthesis Relevance: balanced CV boundary proxy pilot v0.2

## Why v0.2 is needed

v0.1 analyzed 32 targets selected from the first alphabetically sorted subbanks and matched donors by WAV-relative `offset + preutterance`, mistakenly interpreting that coordinate as OTO-relative `preutterance`. Its two-context summary (plain n=16, mean cross-minus-matched +0.113860; rounded n=16, -0.005772) is **not a robust context effect**. In the LynneQing_CVVChn_Mang preliminary run those targets came only from D4/D5 subbanks, and may reflect the biased target composition or incorrect donor matching.

v0.1 remains unchanged for provenance. The v0.2 code is separate and does not write to existing inventories or recording metadata.

## Corrected protocol

- Decode each WAV once and gather existing Mandarin `sh`, `s` or `x` OTO events. The provisional sample coordinate remains `offset + preutterance`, explicitly **not** an independently labeled acoustic CV transition.
- Store **two separate timing values**: `oto_preutterance_ms = entry.preutterance` and `oto_alignment_ms = entry.offset + entry.preutterance`. Match donor candidates on the former and slice audio on the latter.
- Deduplicate identical target WAV/alignment/context events and select one target event per WAV in each context. Allocate up to 16 target WAVs per context by deterministic round-robin over subbanks, with shuffled-like hashed event identifiers preventing alphabetical WAV names from monopolizing a stratum. All subbank allocations and exclusions appear in the result.
- Within the target's subbank and sample rate, donors must have **distinct source WAVs** from one another and the target. Rank them by the absolute difference in OTO `preutterance` duration, with no acoustic score in selection; use at most 12 donors per arm and the same count in both arms (minimum two).
- Compare the original natural adjacent 25 ms windows, median same-context and median cross-context hard-boundary penalties using the existing `splice.boundary_penalty` expression. Positive `cross_minus_same` means the across-context donor's **proxy boundary** has greater discontinuity, not perceptually worse singing.

## Remaining limitations

This is **not** an actual OpenUtau render or listener study. The OTO marking is not guaranteed to sit at the CV transition. Matching preutterance alone does not equate vowel identities, pitch, spectral slope, phonation or recording conditions. The current target cap may leave small subbank strata underrepresented; report counts for each. Source recordings and all OTO metadata remain untouched.

A research finding requires robustness against target caps, subbank stratification, OTO timing sensitivity, phonetic final matching, and independent voices before rendering fixed A/B stimuli and administering blind comparisons. No automatic speaker split/merge recommendations follow.

## Run

```bash
python -m pytest -q
python -m phonoweave.synthesis_relevance_cv_pilot_v02_cli \
  "/path/to/voicebank" --base sh \
  --output ~/Downloads/phonoweave_cv_substitution_pilot_v0.2.json
```
