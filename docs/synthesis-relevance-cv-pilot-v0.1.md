# Synthesis relevance — OTO-aligned CV substitution pilot v0.1

## Scope

This is a new, **read-only** research track separate from adaptive WAV acquisition, speaker-profile generation, and synthesis inventory decisions. It asks whether a context-matched fricative ending is more acoustically continuous with a target's following vowel onset than a fricative ending recorded in another context.

Only Mandarin `sh`, `s`, and `x` plain/rounded groupings are supported. Existing `oto.ini` entries and WAV audio provide the source observations.

## Method

1. Parse the voicebank's OTO aliases, prefix maps and context metadata using the existing Mandarin module. Define a **provisional boundary at WAV-relative `offset + preutterance`**. This is an OTO alignment coordinate, not manually adjudicated fricative–vowel transition.
2. Extract adjacent 25 ms segments immediately before and after this coordinate. Reject truncated windows. Collapse aliases with identical WAV, onset coordinate, consonant, and context.
3. Limit targets to 16 distinct physical WAVs in each context. For each target, find same-context and cross-context donor fricative windows within the *same subbank and sample rate*, excluding every donor from the target's physical WAV. Choose donors deterministically using closeness of OTO preutterance parameter, **never using spectral scores to choose favorable donors**. Use the same number of donors in each arm, at most 12, requiring at least two.
4. Measure `splice.boundary_penalty`, the project's existing sum of log-spectral edge-shape difference plus a weighted RMS discontinuity. Compare the same target's natural contiguous boundary, the median of context-matched artificial boundaries, and the median of cross-context artificial boundaries. The signed difference is `cross - matched`; positive means cross-context substitution has greater **proxy boundary discontinuity**.
5. Retain per-target provenance (relative WAV paths, OTO alias, alignment and hashed event IDs), exclusion reasons, and separate plain/rounded descriptive statistics. Raw WAVs and full voicebank paths are not exported. No automatic `split_recommended` or `merge_supported` actions are produced.

## Interpretation and limitations

**This is not actual synthesized audio.** Each artificial boundary is formed from a donor's 25 ms pre-boundary snippet and the target's 25 ms post-boundary snippet, with no UTAU resampler, pitch conversion, overlap-add, crossfade, F0 matching or full vowel trajectory. OTO `preutterance` can fall inside a consonant or vowel depending on the alias convention. A result is **not evidence that the donor's fricative naturally joins the target vowel** until acoustic boundaries and assembly conditions are independently verified.

A natural contiguous boundary will often have a smaller discontinuity for trivial continuity reasons. The relevant descriptive contrast is **same-context versus cross-context substitution** on each target, not natural versus artificial by itself. Same-context donors can also differ in final identity and pitch; matching their OTO timing is not a phonetic or pitch control. Multiple targets from one singer are not independent speakers.

A follow-up study needs expert-reviewed CV transition positions, actual OpenUtau render A/B stimuli using fixed lyrics, pitch, OTO conditions, and blinded perceptual evaluation. No speaker-profile or synthesis-inventory decisions should be generated from pilot metrics.

## Local command

```bash
python -m pytest -q
python -m phonoweave.synthesis_relevance_cv_pilot_cli \
  "/path/to/voicebank" \
  --base sh \
  --output ~/Downloads/phonoweave_cv_substitution_pilot_v0.1.json
```

The speaker's input source WAV files and OTO annotations are read only.
