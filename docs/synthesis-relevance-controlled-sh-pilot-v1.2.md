# Controlled /sh/ recording pilot v1.2 — preregistered acquisition and integrity gate

## Motivation and scope

The eight-bank v1.1 feasibility audit found **0/92** evaluable pair–pitch-subbank cells, **0** complete four-pair subbanks, and no held-out budget curves. The corpus contained 779 extracted candidate units; candidate OTO units do not equal independent repeated WAVs. All 184 individual-final/subbank slots had either one (141) or two (43) independent WAVs, never the required three. Across banks, 494 independent-WAV counts are bank-local totals, not an estimate of unique global speakers or sessions. No split-policy performance conclusion follows.

**Research decision:** do not lower the v1.1 requirement (>=2 different training WAVs plus >=1 held-out WAV per final) to manufacture results. Establish a controlled pilot first; subsequent comparative evaluation remains a separate stage.

## Frozen pilot acquisition grid

- One consenting pilot speaker and one comfortable, stable sung pitch region, with a documented target fundamental frequency. Pilot scope **does not establish speaker generalization**.
- Eight Mandarin /sh/ finals exactly: `sha/shua`, `shai/shuai`, `shan/shuan`, `shang/shuang` (orthographic prompts with four paired finals `a/ua`, `ai/uai`, `an/uan`, `ang/uang`). The pairs are research controls, not claims of identical phonetic nuclei.
- Two independently captured sessions on **different calendar dates**, with recording setup explicitly reset between sessions. Session S01 is **train**; S02 is **heldout**. Each session has three independently recorded WAV takes for each of eight finals: **24 files per session, 48 total**. An independent take entails a fresh production, not copying, trimming, or re-encoding an existing file.
- Keep singer, microphone, acoustic location, microphone position, target pitch, gain and vocal register consistent as practicable; record session device, room, gain and pitch metadata, including any departures. Record the assigned target from a fixed randomized playback-free prompt order or log the actual order separately to help diagnose ordering confounds. Do not normalize/edit audio in place.
- Save mono uncompressed PCM WAV, ideally 48 kHz/24-bit (the audit accepts 44.1/48/96 kHz, PCM 16/24 bit but requires uniform rate/depth). Capture at least 300 ms and at most 8 s of signal per file, including pre/post phonation where possible. Avoid clipping, excessive background noise and unstable phonation; these require additional **manual quality review**, beyond basic integrity checks.
- Label onset of /sh/ frication and boundary of the subsequent glide/vowel transition independently, in milliseconds from file start, using a frozen annotation rubric with inspection notes and reviewer IDs retained separately. Automatic boundary extraction without an independent check is insufficient to claim verified frication-only signatures.

## Manifest and integrity audit

Generate a manifest before recording:

```bash
python -m phonoweave.synthesis_relevance_controlled_corpus plan \
  --output ~/Downloads/controlled_sh_pilot/manifest.json
```

This creates the complete immutable **48-take target grid** with expected relative filenames under `audio/S01` and `audio/S02`. Fill in both sessions' capture date (YYYY-MM-DD), device ID, room ID, gain setting, target pitch Hz and `setup_reset_confirmed=true`. For each take, add the **actual SHA-256** of the captured WAV and manually reviewed `frication_onset_ms` and `cv_boundary_ms`. Never change the predeclared pairs, roles or counts after capture. Store consent records securely outside the repository; share only approved audio.

Audit after recording and annotation:

```bash
python -m phonoweave.synthesis_relevance_controlled_corpus audit \
  ~/Downloads/controlled_sh_pilot/manifest.json \
  --root ~/Downloads/controlled_sh_pilot \
  --output ~/Downloads/controlled_sh_pilot/audit.json
```

The auditor rejects absent files, changed grid/roles, duplicate file hashes, unapproved/heterogeneous format, out-of-range or missing annotations, omitted provenance/hash metadata and same-date sessions. Exit code 2 means **not analysis-ready**, and no effect is estimated. Hashing alone cannot detect perceptual duplicates or hidden source overlap; human provenance review is required. Onset labels are recorded but not yet consumed by the frozen v1.1 proxy extraction.

## Evaluation stages and safeguards

1. **Capture/integrity gate:** verify 48 separate WAVs, valid metadata, independently labeled boundaries, manual audio QC. Until all conditions pass, status is insufficient, not a null policy effect.
2. **Interface/construct validity gate:** verify /sh/ OTO mappings and context extraction on pilot audio, duration-matched feature regions and label quality. The existing `extract` depends on actual OTO/WAV voicebank structure; this manifest is **not yet a drop-in input** to v1.1. An independent adapter plus its leakage tests must be implemented before any curves are generated.
3. **Session-heldout proxy comparison:** freeze S01 train and S02 test, rank adaptive choices using S01 only, compare equal budgets 0–4 with the same target recordings. Endpoints 0 and 4 must coincide; only budgets 1–3 can distinguish policies. Report per-pair coverage, decisions, losses and uncertainty; do not optimize thresholds on S02.
4. **Confirmatory extension (separate protocol):** multiple independently recruited speakers, more sessions, stronger out-of-speaker baseline and actual synthesis-resampler A/B quality assessment with equal **measured recording cost**. Recording count is not identical to wall-clock acquisition cost. A pilot advantage on offline log-spectrum is not evidence of rendered quality.

## Nonclaims and stop conditions

This is a **data-acquisition pilot**, not validated synthesis or proof that adaptive splitting improves quality. Do not mix speaker identities or pitch subbanks as though recordings were interchangeable. Do not use S02 to choose acoustic feature settings or policy ranks; if it becomes development material, recruit an untouched third evaluation session. A missing cell, clipping, unverifiable provenance or non-independent repeats blocks the affected comparison instead of being treated as zero loss.

The artifact contains no original audio and does not modify legacy WAV/OTO. The preceding draft PR #54 remains separate.

## Optional guided terminal capture (macOS)

The non-overwriting, resumable microphone recorder has an optional local audio input dependency:

```bash
python -m pip install sounddevice
python -m phonoweave.synthesis_relevance_controlled_capture --list-devices
```

After microphone, clipping level, stable /sh/ production, pitch target, and session-reset checks, acquire S01:

```bash
python -m phonoweave.synthesis_relevance_controlled_capture \
  --manifest ~/Downloads/controlled_sh_pilot/manifest.json \
  --session S01 --room-id quiet_room --gain-setting fixed \
  --target-pitch-hz 220 --confirm-setup-reset
```

220 Hz is only an example, not a prescribed pitch. Select a comfortable, measurable target and preserve it in S02. Use `--device` if the default microphone is incorrect. The utility captures 48-kHz mono PCM16 WAVs using fixed filenames. Enter starts a take; q quits. Re-running verifies recorded hashes and resumes safely. Files without a matching manifest hash are never overwritten. S02 is blocked until S01 hashes are complete and must be recorded on a later date. Do not change room, microphone, gain, or target pitch during a partial session.

**Acquisition limitations:** no auditory count-in, no internally generated pitch reference, no calibrated clipping/volume or pitch-stability check, and no built-in independent perceptual QC. The recorder detects stream overflow; the capture begins after the Enter trigger, so reaction time is included in the recording. Microphone name is OS metadata, not hardware attestation. Take order is deterministically shuffled per session; the order index is saved with each file. Onset/CV boundary labels remain separate manual annotations, so the audit correctly stays not-analysis-ready after raw capture.
