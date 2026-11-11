# PhonoWeave v18 — single-command, bounded recording plan workflow

## Why this exists

PhonoWeave's purpose is to inspect how a singer's existing recordings cover synthesis-relevant pronunciations and recommend **only necessary diagnostic recording material**. v16 added line limits; v17 counted exact distinct PCM rather than treating every OTO alias as a new audio source. v18 **composes** those established functions into a single preview/export command. It is not a new acoustic model, a singer, or a standalone phoneme detector.

## Input and behavior

Pass the path to an **already extracted existing voicebank directory** containing `oto.ini` and referenced WAVs. The current diagnostic candidate generator handles Mandarin orthographic syllables only; arbitrary user spellings and other languages remain outside this workflow and must not be silently interpreted as Mandarin.

```bash
# Preview only (recommended first)
phonoweave plan-recording /path/to/voicebank --max-lines 8

# Skip lines already planned; then create an entirely new export folder
phonoweave plan-recording /path/to/voicebank \
  --max-lines 8 \
  --existing-reclist /path/to/my_previous_reclist.txt \
  --export /path/to/new_output_folder

# Machine-readable preview without creating any files
phonoweave plan-recording /path/to/voicebank --json
```

A successful `--export` creates **two** files outside the voicebank:

- `supplement_reclist.txt`: only newly recommended diagnostic lines (may be empty).
- `recording_plan.json`: source inventory, recognized coverage, evidence gaps, why each proposed recording was selected, source-aware PCM counts, skipped/deferred lines and limitations.

No export path means **no writes**, no OTO, voicebank, alias or WAV changes. An existing export directory, one nested inside the source voicebank, or one whose parent does not exist, is rejected. Existing files are never overwritten. The output JSON contains the local source path; review it before sharing.

## What a report means

- `recognized_syllable_observations` counts a subset of **Mandarin-readable, WAV-backed OTO aliases**, not independent performances or verified phone boundaries.
- `pcm_wav_paths_fingerprinted` counts decoded PCM paths; exact identical PCM is de-duplicated in the v17 selector, but two different PCM digests are not proof of distinct performances.
- `evidence.gaps[].next_action` distinguishes `supplemental_recording` from other validation work (e.g. perceptual validation). Only actual supplemental recording requests generate lines.
- `recording_plan.already_scheduled` is **planned work**, not existing acoustic evidence. `max_lines` bounds script lines, **not the number of pitch takes**.
- An empty recommendation can mean no eligible gap, insufficient evidence or already scheduled lines. It **must not** be interpreted as complete acoustic coverage.
- The core speaker-inventory analyzer remains unchanged. No automatic split/merge or new synthetic unit is approved by this command.

## Error states and scope safeguards

A missing/non-folder path, absent usable OTO records, no existing referenced WAV, or zero recognized Mandarin observations is rejected with a direct message. Missing or unreadable individual WAVs and unimplemented onset types appear as warnings when analysis remains possible.

The workflow intentionally does not accept ZIPs, produce a full-length singing recording program, support user-native non-Mandarin phoneme notation, or modify the graphical interface. Those require separate input contracts, language modules and usability validation, not assumptions that would fabricate coverage.

## Engineering tests

`tests/test_recording_workflow_v18.py` checks preview, export, overwrite and source-tree protection, invalid inputs, budget bounds, warning preservation, call-chain reuse and a tiny genuine PCM16-WAV/OTO integration fixture (with the expensive inventory analysis deliberately stubbed to a known evidence gap; all other processing remains real in CI). Running `pytest -q` in GitHub tests the combined repository and existing commands. These software checks do not establish synthesis accuracy or actual minutes saved.