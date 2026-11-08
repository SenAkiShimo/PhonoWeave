# Independent 36-take VC pilot: an acquisition gate, not another split heuristic

## Rationale

LynneQingD4's six annotated `i/ir/i0 -> m/sh` excerpts diagnose plausible acoustic differences and a misleading OTO-centered 16 ms feature, but provide **one excerpt per condition**, partly from different source WAVs. Further mining the old recordings cannot estimate within-context and between-session variation or establish synthesis benefit. This milestone therefore does **not** score, split, train or render from those six examples. It prepares the *new recordings required to falsify* the proposed speaker-adaptive benefit.

Rather than invent a new microphone recorder, this is a **DAW-friendly, reproducible ingest and audit**. Record in Audition, Reaper, RØDE software, etc.; WAVs are the evidence. The original voicebank and v0.7 annotations are never modified or uploaded.

## Protocol, frozen before S01

- Same singer in each session. Singing the old D4 voicebank *is not mandatory*: using a different singer creates a new single-speaker study, not replication of Lynne's results.
- Six targets: `li ma` / `shi ma` / `zi ma` and `li sha` / `shi sha` / `zi sha`, each 3 **fresh independent takes** in each of S01/S02 (18 + 18 = 36).
- S01: training/model-choice only. S02: different *later* calendar day, held out from choosing acoustic features, split decision thresholds or rendering parameters.
- Maintain the same singer, pitch, duration, note lengths, mic/gain/room/distance. All syllables are level-tone laboratory elicitation prompts, not claims about natural lexical prosody. Fix pitch/score for singer comfort **before** capture and do not alter afterward.
- Each WAV contains the *full two-syllable continuous utterance*. Do not copy or duplicate fragments, use pitch correction, reverb, compression, or other processing. Keep dry mono PCM16 audio at a **consistent** 44.1, 48, or 96 kHz. Exact matching source audio byte hashes and decoded PCM hashes are rejected.
- Manually confirm one transition onset/offset in each recording; each annotation must cite exact source WAV SHA-256. Any capture QC flags require manual review. Human metadata does not independently prove session dates or absence of leakage.

## Commands on this branch

```bash
phonoweave vc36 plan --manifest /experiment/vc36/manifest.json \
  --speaker-id speaker_pilot_01 --pitch-hz 293.66 \
  --duration-s 3 --first-note-ms 750 --second-note-ms 750

phonoweave vc36 cues --manifest /experiment/vc36/manifest.json \
  --output /experiment/vc36/cue_sheet.csv
```

Use the cue sheet's per-session randomized prompt order; record 18 separate WAVs for S01 and 18 for S02. For each session, export DAW WAVs with exact `take_id.wav` filenames into a **separate external** folder, such as `/captures/S01`. Export using the same dry sample rate/bit depth throughout. A partial session can be ingested and resumed safely.

```bash
phonoweave vc36 ingest --manifest /experiment/vc36/manifest.json \
  --source /captures/S01 --session S01 --date 2026-11-15 \
  --device-id rode_microphone --room-id desk_room --gain-setting fixed \
  --distance-cm 18 --confirm-reset

# On a *later day*, after all 18 S01 WAVs have been ingested:
phonoweave vc36 ingest --manifest /experiment/vc36/manifest.json \
  --source /captures/S02 --session S02 --date 2026-11-17 \
  --device-id rode_microphone --room-id desk_room --gain-setting fixed \
  --distance-cm 18 --confirm-reset
```

Run `vc36 audit` before asking the assistant to compare anything. It will report `capture_complete` only after all WAVs, SHA-256 and session information pass; this is **not** the same as `analysis_ready`.

```bash
phonoweave vc36 audit --manifest /experiment/vc36/manifest.json \
  --output /outside/first_audit.json
```

For manually annotated transition intervals, provide a CSV with **exactly** these columns:

`take_id,source_wav_sha256,transition_onset_ms,transition_offset_ms,boundary_reviewer,qc_manual_reviewed`

Each row must cite the take ID and SHA from the verified manifest, and use timestamps measured from the *beginning of that take's WAV*. A reviewer may append a batch of a few completed rows; already-reviewed boundaries cannot silently be replaced. Incomplete/unreviewed rows remain unresolved.

```bash
phonoweave vc36 annotate --manifest /experiment/vc36/manifest.json \
  --csv /outside/reviewed_boundaries.csv
phonoweave vc36 audit --manifest /experiment/vc36/manifest.json \
  --output /outside/final_audit.json
```

All audit JSON outputs are newly created **outside** the original audio directory; `ingest` copies to new canonical slots exclusively and never overwrites an existing WAV. `plan`/`cues`/`audit` refuse to overwrite existing files.

## What this specifically does NOT prove

Successful ingestion proves only that the files satisfy the declared acquisition protocol; it does not establish accurate vowels, independent phonations, pitch equivalence, annotation correctness, stable acoustic effects, or perceptual gain. The `analysis_ready` flag means *eligible for evaluation*, **not** any approval to split/share.

Next scientific test (requires these real recordings): use the **same 18 S01 source takes** as input to (a) phoneme/context-aware shared pooling and (b) speaker-adaptive context partitioning. Train/fit using S01 only, freeze thresholds and rendering configurations, then evaluate all 18 S02 references with equal acquisition cost. The shared baseline must be a genuine pooled estimator — never the older cross-context substitution used for pipeline smoke tests. Report acoustic and rendering quality; include blind listening when feasible.

One singer is only a feasibility pilot, not a generalizable multilingual claim. If the singer cannot be re-recorded, use a new consented singer and disclose the difference; do not reuse old WAVs as if they were new independent takes.

## Testing

```bash
python -m pytest -q tests/test_vc36_capture.py
```

Tests use synthetic PCM WAVs solely to validate the software gates; they are not real PhonoWeave speech experiment results.
