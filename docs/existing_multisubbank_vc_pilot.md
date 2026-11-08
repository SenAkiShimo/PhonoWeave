# PhonoWeave: retrospective DiffSinger-generated, seven-subbank CVVC pilot

## Source provenance: confirmed by voicebank author

The author confirmed **all provided pitch subbanks were generated directly with DiffSinger**, rather than independently sung as human takes. There are two designated voice styles (Mang/芒, Lu/露) and seven pitch subbanks. Do not count these as independent human recordings, independent singers or distinct DiffSinger model checkpoints. The exact DiffSinger model checkpoint(s), inference seeds, shared note/phoneme/duration inputs, and timbral conditioning controls are **not established** by the ZIP files or this confirmation.

This pilot's target population is **DiffSinger-generated audio for subsequent concatenative voicebank construction**. It is useful for synthesis-to-concatenation adaptation and controlled pitch/style stress tests, not direct inference about natural singer coarticulation.

## What was audited

The prior Mang D4 bank was combined with four newly supplied ZIP archives:
Mang G3 / D4 / G4 / D5 and Lu C4 / B4 / F4. The integrated Lu archive also contains a separate `sample.wav` (excluded from recording counts). The pilot reads 175 ordinary WAV files at the structural level; it decodes and measures six selected VC candidates in each of the seven pitch subbanks (42 candidate excerpts). No audio is changed, extracted into the public report, or uploaded to GitHub.

## Most important factual result: copied OTO time templates

After removing the pitch prefix, the exact source WAV / alias / five OTO time-field tuple matches Mang D4 for 99.1% to 100% of rows per other subbank. Across the 25 common WAV names, the median seven-bank duration spread is **8.48ms** and the maximum is **11.38ms**. This is a highly synchronized timing layout consistent with shared DiffSinger conditioning or export templates, but the precise cause has **not** been established. The decoded PCM hashes of the source WAVs used for our 42 candidate excerpts are *different* across banks. Different hashes do not prove independent takes or different generative seeds, because the renderer may differ in pitch/style settings.

## Pilot: six VC labels per pitch

All 7 subbanks contain the same six alias/WAV targets: i/ir/i0 followed by m or sh. This yields 21 /sh/-shaped and 21 /m/-shaped source-linked snippets.

- In all 21 /sh/ cases, a median high-frequency (2.2–8kHz) energy fraction changes markedly from pre-anchor (-60 to -20ms) to post-anchor (+40 to +80ms). The weakest recorded increase is **+0.861**.
- In the 21 /m/ cases, none shows a positive increase exceeding +0.01 under this same friction-specific diagnostic.

This **supports the gross distinction between /sh/-like frication and /m/-like nasal transitions at the shared time grid**. It does not validate the location of the exact phonetic boundary in the six new banks, establish that each recording is independent, or demonstrate speaker-adaptive split vs share. In particular, the `ir sh` examples are taken from `ingi0ir.wav` whereas `i sh` and `i0 sh` come from `vcsection3.wav`.

## Critical negative/uncertain result

For each pitch and right-hand context, we compared five equal normalized positions inside the D4-human-annotated interval. For other banks these intervals were *transferred by OTO anchor* and are **not human-validated**. Log-band distance ranks among i/ir/i0 are **not stable** across all subbanks. All three possible nearest pairs occur across the 7 subbanks for both /m/ and /sh/. Thus **the data do not support one fixed “share these two labels” decision**. Differences may be due to pitch or timbral conditioning, generator behavior, spectral tilt, exact timing and transferred boundary error, not just phoneme identity. This is descriptive pilot data, not an inferential test.

## Decision and next research step

**GO:** the existing multi-pitch audio is valuable as a retrospective *stress test* for representation and segmentation. **NO-GO:** treat these seven subbanks as seven independent replications, report a split/merge efficacy effect size, tune policy on all subbanks and then call one held-out.

Production method is now confirmed as **DiffSinger direct rendering**. Do not request 36 human re-recordings for this synthetic-audio pilot.

**Next decisive experiment (frozen render benchmark):** take an unchanged set of DiffSinger source WAVs as the *only* allowed unit inventory, compare two genuinely executable concatenative renderers/policies—(A) preserve author-defined separate VC classes and (B) selectively pool/share candidate VC classes—under matched source-sample or recording-duration budgets. Keep pitch, voice style, downstream note targets, resampler and wavtool fixed per pair. Evaluate on **unseen syllable sequences** with separately rendered DiffSinger reference vocals; do not merely replay or splice a training-reference passage. Freeze source/target membership before optimizing policy; use actual objective continuity/alignment and blinded listening tests. In this setting, DiffSinger references are a **teacher-generated target**, not human ground truth. The answer is whether unit reuse preserves teacher-output quality, *not* whether it reproduces natural human singer behavior.

Before a strong claim, confirm whether bank generation reused the same MIDI/phoneme duration templates and whether Mang/Lu share any underlying model/checkpoint; this matters for generalization and train-test contamination. Manually check representative non-D4 boundaries before spectral policy learning. The 36-take human-session protocol remains a distinct, optional later external-validation study.

## Reproduce

Keep all five source ZIPs next to a local manifest. Paths in the manifest can be relative.

`python multisubbank_vc_audit.py --manifest bank_manifest.example.json --output output-new`

`pytest -q test_multisubbank_vc_audit.py`

Outputs (`summary.json`, `subbank_inventory.csv`, `vc_target_features.csv`, `pairwise_acoustic_shapes.csv`, `wav_duration_spread.csv`, graphs) include source archive SHA-256 and OTO hashes but no original audio. The `source_PCM_sha256` fields let the author detect accidental waveform reuse.
