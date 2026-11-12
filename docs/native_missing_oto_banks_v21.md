# PhonoWeave v2.1 — do not overlook unconfigured subbanks

## Why this matters

v2.0 scanned only subbanks with an `oto.ini`. If a singer had an audio-only pitch bank (for example, `D4/ŋa.wav` but no `D4/oto.ini`), it could be excluded from template checks and yield an incorrect impression of complete coverage.

This **small corrective patch** keeps the user-native template workflow and does **not** invent sounds, IPA names, or a new acoustic model.

## Change

- Explicit subbank declarations supplied with `--native-bank` or via a target JSON manifest's `subbanks` list are accepted even when that existing subbank directory has no `oto.ini`.
- Such declared subbanks are checked for the exact WAV filename and any usable OTO association. An existing readable WAV without registration becomes `audio_present_needs_oto_review`, **not** a request to rerecord. Missing or invalid WAV remains a recording gap, with invalid audio distinguished in each bank's status.
- Output includes `voicebank.subbanks_missing_oto` and a warning. This is an administrative repair hint; no `oto.ini` is synthesized or overwritten.
- Declared subbank paths must exist and resolve within the voicebank. Unspecified directories without OTO are **not** auto-classified as pitch subbanks, avoiding false positives from caches, downloads, and auxiliary WAVs.
- If no subbank is discoverable or declared, fail with an explanatory error instead of returning a falsely complete plan.
- Retains v2.0's exact user-defined labels, bounded recording budget, non-destructive exports, and existing Mandarin mode.

## Example

```bash
phonoweave plan-recording /path/to/voicebank \
  --native-reclist /path/to/original_reclist.txt \
  --native-bank D4 --max-lines 8
```

Or supply a user-native JSON manifest (schema `phonoweave.user_native_recording_targets.v1`) with per-target `subbanks` declarations.

For `D4/ŋa.wav` with no `D4/oto.ini`, the plan should recommend OTO review, **not a second recording**. If the target file is missing, it can recommend recording `ŋa` in D4.

## Validation limits

Tests include audio-only and empty subbanks, explicit flags and manifest scopes, missing declared directory failure, and backward compatibility with a registered OTO bank. These are synthetically generated PCM WAVs and are **not proof** of general phoneme discovery, recording savings, or correct acoustic-unit identification.

The project remains focused on **speaker-specific sound realization, justified recording coverage, and minimal reclist generation**. No voicebank data or research archives are committed.
