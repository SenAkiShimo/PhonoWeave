# v1.2 — Independent DiffSinger teacher intake and matched-budget diagnostics

## Why this milestone exists

PR #71 generates 56 actually audible note-timed, pitch-shifted Rubber Band concatenative **previews** for seven pitch subbanks and two voice styles. All share one LynneQing DiffSinger voicebank, originally directly rendered into author-designed CVVC unit WAVs. The current ZIPs are source *units*, not model files. They contain no accessible DiffSinger `.ckpt`/`.onnx` inference model. No independently generated teacher vocals exist in the supplied data.

Therefore this module does **not** synthesize an artificial teacher by concatenating stored units, does **not** treat a previous preview as reference, and generates **no** comparative quality metric unless a separate operator-attested reference WAV passes all checks. The model/project still has to be provided or a real DiffSinger editor must export the same frozen score.

## Existing fixed protocol

- Source preview: `render_manifest.json` from PR #71, 56 **real** WAVs, SHA-256 hashes, frozen note score and 14 Standard MIDI files.
- Paired conditions: only three K2 policies—`k2_keep_i_ir`, `k2_keep_i_i0`, `k2_keep_ir_i0`—share **2,140 ms** of unique source per pitch. `k3_preserve_all_not_budget_matched` costs **2,380 ms**; exclude it from matched-budget statistics.
- Seven banks: Mang `G3/D4/G4/D5`, Lu `C4/F4/B4`. These are styles of **one DiffSinger voicebank**, not independent speakers.
- Score: two six-note phrases per subbank, encoded in the unchanged MIDI and note/lyric table. Note that MIDI lyrics and note lengths alone may **not** uniquely determine a DiffSinger phoneme-duration/F0 inference run.

## CLI

```bash
# 1. Verify all 56 previews/MIDI and create a 14-row *unfilled* external reference receipt.
phonoweave diffsinger-teacher prepare \
  --renders /private/LynneQing_NoteTimed_VC_AB_v11 \
  --output /private/reference_intake.json

# 2. Independently render the exact matching phrases in DiffSinger and place
#    each mono PCM16 44.1kHz WAV in teacher/<bank>/<phrase>.wav.
#    For a first feasibility pilot, try ONLY:
#       teacher/mang_D4/phrase_a.wav
#       teacher/lu_C4/phrase_a.wav
#    Use corresponding score_midi/<bank>_phrase_a.mid from the frozen score.

# 3. Populate only file hashes (does NOT attest they are genuine DiffSinger).
phonoweave diffsinger-teacher scan \
  --receipt /private/reference_intake.json \
  --reference-root /private/teacher_collection \
  --output /private/reference_scanned.json

# 4. For the captured target rows, manually confirm in reference_scanned.json:
#    model_or_checkpoint_identifier = a meaningful identifier
#    settings_or_project_reference = a stable project/parameter identifier
#    new_generation_from_frozen_score = true
#    source_wav_reused_or_spliced = false
#    style_verified_by_operator = true
#    note_and_lyric_sequence_verified = true
#    onset_alignment_verified = true
#    onset_shift_ms = 0 or explicitly human-reviewed timing offset
#    These are OPERATOR ATTESTATIONS; the software cannot infer them from WAV bytes.

# 5. Audit and compare verified rows only; others remain blocked.
phonoweave diffsinger-teacher audit \
  --renders /private/LynneQing_NoteTimed_VC_AB_v11 \
  --receipt /private/reference_attested.json \
  --reference-root /private/teacher_collection \
  --output /private/teacher_diagnostics.json
```

## Comparison controls and measurement

Each admitted reference must have the frozen MIDI hash and note-score SHA-256; a WAV file SHA-256; an identified DiffSinger model/version or checkpoint; operator-confirmed style, lyrics and score; a new synthesis run with no source ZIP waveform reuse; and verified sample-rate/timing alignment. Any exact match with a K2/K3 preview is blocked. Exact output sample rate must equal each bank's previous Rubber Band render (currently 44.1kHz), and phrase duration must agree to within 100ms; otherwise align/convert externally and reverify.

For each approved reference phrase, the module measures two descriptive features versus the *same* teacher signal and times: 12-band normalized log-power spectral shape framewise RMSE (32ms Hann frame, 10ms stride, active teacher frames only), and absolute framewise RMS difference in dB. It never independently fits each arm's time shift or gain to its own teacher. The K3 arm is never scored as a matched-budget comparator. Any preliminary ranking is about numerical reconstruction error only; it is **not** demonstrated perceptual quality, pronunciation fidelity, natural singing fidelity or phonetic split/merge benefit.

## Why the benchmark is still limited, even after teacher audio

The six VC context types in these phrases already exist in the source inventory. Even though continuous phrases may be new, this is not an unseen-phoneme or cross-speaker test. Same underlying DiffSinger voicebank and its synthetic output can introduce dependency through generator templates and style controls. The current renderer uses Rubber Band DSP and hand-cut OTO-relative unit excerpts rather than a validated OpenUtau resampler/wavtool. True synthesis improvement will require stricter held-out sequence selection, external renders from a frozen vocal synthesis workflow, more independent conditions and blinded intelligibility/quality ratings.

## Verification

Nine synthetic software regression tests (including positive fake teacher, negative missing teacher, cheating by using preview WAV, forged/missing attestation, score corruption, audio corruption, unsafe paths, hash scanner without auto-attestation) passed locally. Real-preview audit: **0/14 external teacher WAVs present → 0 reconstruction scores**, as required. No real DiffSinger teacher audio has been generated by this software.
