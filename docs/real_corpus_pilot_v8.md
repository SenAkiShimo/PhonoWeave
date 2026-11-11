# PhonoWeave v8 — real archive preflight and blinded pilot intake

This preflight does **not** discover phones, score segmentation, or require IPA. It binds real frozen v3/v4/v5 artifacts, checks source identity across v4/v5, and generates a deliberately incomplete independent-script template.

## Source data and selection

- Two subbanks: `mang_D4`, `lu_C4` (not independent speakers).
- Retrospective template-heldout pilot: `e0.wav` and `en0.wav` from each bank (4 files). These groups were held out in v3 but were exposed to **posthoc OTO diagnostics in v5**; they are not pristine confirmatory data.
- Workflow-only smoke: `additional.wav` from each bank (2 files, v3 development groups). Never mix smoke metrics into heldout results.
- Selection is fixed in code before reviewing any actual manual boundaries. No OTO/IPA or inferred transcription used.

## Commands

```bash
python -m phonoweave.native_pilot_v8 prepare \
  --v2 /path/PhonoWeave_User_Native_Batch_Discovery_v2.zip \
  --v3 /path/PhonoWeave_User_Native_Temporal_Context_v3.zip \
  --v4 /path/PhonoWeave_WholeAudio_Unanchored_Events_v4.zip \
  --v5 /path/PhonoWeave_UserNative_EventLink_v5.zip \
  --output /path/NEW_v8_preflight
```

Output: `cohort_frozen.json` with source identities and original archive SHA-256, `cohort_sha256.txt`, `preflight_report.json`, `sequences_TEMPLATE.json`, and `audio_manifest_TEMPLATE.json`.

**The sequences template is intentionally unusable** until completed by a human with the *complete ordered native-unit list per WAV*, independently obtained from a known original recording script (not OTO, not v5 predictions, not filename guesses). Set `provenance` to `independent_recording_script` only when that is genuinely how the list was obtained. `verify-sequences` checks schema and hash/source consistency, not truth of the asserted provenance.

```bash
python -m phonoweave.native_pilot_v8 verify-sequences \
  --cohort /path/NEW_v8_preflight/cohort_frozen.json \
  --sequences /path/filled_seqs.json
```

To extract audio safely for *local* annotation, fill in the real ZIP paths in the audio manifest (the ZIP-relative roots come from the frozen v2 manifest, never its private archive paths). All original archive SHA and WAV PCM hashes must match before extraction.

```bash
python -m phonoweave.native_pilot_v8 extract-audio \
  --cohort /path/NEW_v8_preflight/cohort_frozen.json \
  --manifest /path/local_audio_manifest.json \
  --output /path/NEW_private_audio
```

**Never upload original voicebank audio to the public GitHub repo** without permission. The software doesn't need a phonetic dictionary or IPA. Filename categories are not complete pronunciation sequences.

Once independent scripts exist, run v6 alignment, v7 freeze, then prepare BLIND manual annotations (annotator sees original audio and scripted unit order, not model predictions), followed by v7 scoring. External validation needs *new recordings* from independently held-out capture conditions and speakers; these existing files are only retrospective feasibility checks. For coarticulation, predefine manual onset/offset conventions and include an explicit uncertain label rather than force certainty.