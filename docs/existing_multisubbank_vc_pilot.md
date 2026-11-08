# PhonoWeave: retrospective seven-subbank CVVC pilot

## What was audited

The prior Mang D4 bank was combined with four newly supplied ZIP archives:
Mang G3 / D4 / G4 / D5 and Lu C4 / B4 / F4. The integrated Lu archive also contains a separate `sample.wav` (excluded from recording counts). The pilot reads 175 ordinary WAV files at the structural level; it decodes and measures six selected VC candidates in each of the seven pitch subbanks (42 candidate excerpts). No audio is changed, extracted into the public report, or uploaded to GitHub.

## Most important factual result: copied OTO time templates

After removing the pitch prefix, the exact source WAV / alias / five OTO time-field tuple matches Mang D4 for 99.1% to 100% of rows per other subbank. Across the 25 common WAV names, the median seven-bank duration spread is **8.48ms** and the maximum is **11.38ms**. This is a highly synchronized timing layout; it is not evidence of independent repeated phonation. The decoded PCM hashes of the source WAVs used for our 42 candidate excerpts are *different* across banks. Different hashes do not prove independent takes, because signal processing can also alter PCM.

## Pilot: six VC labels per pitch

All 7 subbanks contain the same six alias/WAV targets: i/ir/i0 followed by m or sh. This yields 21 /sh/-shaped and 21 /m/-shaped source-linked snippets.

- In all 21 /sh/ cases, a median high-frequency (2.2–8kHz) energy fraction changes markedly from pre-anchor (-60 to -20ms) to post-anchor (+40 to +80ms). The weakest recorded increase is **+0.861**.
- In the 21 /m/ cases, none shows a positive increase exceeding +0.01 under this same friction-specific diagnostic.

This **supports the gross distinction between /sh/-like frication and /m/-like nasal transitions at the shared time grid**. It does not validate the location of the exact phonetic boundary in the six new banks, establish that each recording is independent, or demonstrate speaker-adaptive split vs share. In particular, the `ir sh` examples are taken from `ingi0ir.wav` whereas `i sh` and `i0 sh` come from `vcsection3.wav`.

## Critical negative/uncertain result

For each pitch and right-hand context, we compared five equal normalized positions inside the D4-human-annotated interval. For other banks these intervals were *transferred by OTO anchor* and are **not human-validated**. Log-band distance ranks among i/ir/i0 are **not stable** across all subbanks. All three possible nearest pairs occur across the 7 subbanks for both /m/ and /sh/. Thus **the data do not support one fixed “share these two labels” decision**. Differences may be due to pitch, voicing style, exact timing, session or microphone, not just phoneme identity. This is descriptive pilot data, not an inferential test.

## Decision and next research step

**GO:** the existing multi-pitch audio is valuable as a retrospective *stress test* for representation and segmentation. **NO-GO:** treat these seven subbanks as seven independent replications, report a split/merge efficacy effect size, tune policy on all subbanks and then call one held-out.

Before any stronger test, obtain the bank-production provenance: were pitch subbanks independently sung, time-aligned to one guide track, pitch-shifted, or derived by voice conversion? This matters because the WAV durations and OTO time fields are extraordinarily synchronized.

With provenance clarified, verify boundaries on selected new subbanks (a small targeted human check, not thousands of rows). Then compare actual resampler synthesis under equal available recording costs and frozen evaluations. New 36-take recording protocol is *optional*, not a prerequisite to this retrospective audit.

## Reproduce

Keep all five source ZIPs next to a local manifest. Paths in the manifest can be relative.

`python multisubbank_vc_audit.py --manifest bank_manifest.example.json --output output-new`

`pytest -q test_multisubbank_vc_audit.py`

Outputs (`summary.json`, `subbank_inventory.csv`, `vc_target_features.csv`, `pairwise_acoustic_shapes.csv`, `wav_duration_spread.csv`, graphs) include source archive SHA-256 and OTO hashes but no original audio. The `source_PCM_sha256` fields let the author detect accidental waveform reuse.
