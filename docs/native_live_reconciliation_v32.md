# PhonoWeave v3.2 — recording-plan / live WAV-OTO consistency check

## Product objective

Directly compare the **current voicebank's** author-supplied native recording targets with the current `oto.ini` to WAV/PCM source links, in the same Recording Plan preview. Avoid unnecessary rerecording while preserving actual missing source-file warnings. No acoustic classifier or new phonetic unit is added.

## How to use

Select a voicebank directory, supply your existing native recording template (`reclist.txt` or target JSON), and click **分析并预览**. v3.2 performs the original native planner followed by a read-only live file/OTO scan. The same preview now displays a file-consistency section with file-level agreements, open reviews/missing sources, and possible contradictions.

The existing CLI `phonoweave plan-recording VOICEBANK --native-reclist original_targets.json --max-lines 8 --json` also adds a `source_consistency_review` section. Normal export includes it within `recording_plan.json` alongside the original `supplement_reclist.txt` and `recording_review.txt`; it does not introduce a new rec-list of potential experiments.

## Conservative cross-check rules

- `source_verified_file_level`: a requested alias has the same accessible, decoded PCM WAV in both planner and live scan. This proves file/registration agreement, **not correct phoneme realization or OTO segment timing**.
- `source_file_missing`: OTO explicitly points at a missing original WAV, and both scans agree. This is a missing **file source**, not proof of a never-performed vocal unit.
- `source_repair_review`: WAV exists but cannot be decoded; check backups before requesting rerecord.
- `source_mapping_review` / `oto_or_source_review`: insufficient or mismatched alias/OTO registration, including long-WAV naming and cross-subbank references.
- `source_choice_review`: more than one original WAV is linked to an alias; neither source is automatically chosen.
- `possible_inconsistency`: planner and live source evidence disagree, possibly because voicebank files were edited between the two sequential scans. Flag it for investigation, not an automatic recording task.

All original author labels and bank prefixes remain opaque. An accepted `i`, `ir`, or `i0` is never automatically merged or split. Uncertainty about whether `a`, `ei`, or `hw` are suitable new recording contexts remains unchanged.

## End-to-end test

The regression fixture creates a realistic long-WAV, multi-alias subbank alongside a missing OTO-registered WAV, an unreadable WAV, an ambiguous two-WAV alias, and a WAV without registration. It checks that the **same** source is verified by both paths, that conflicting statuses are visible, and that the generated supplement and source-review files contain no unintended new lines. Additional checks simulate a source file disappearing between scans and exercise the local GUI endpoint.

**Limitation:** The full original LynneQing WAV+OTO voicebank is not available in the current workspace; therefore this is synthetic-fixture integration acceptance, not real-corpus acoustic or functional accuracy validation. This does not measure recording-time reductions or independently performed takes.

Source WAV/OTO files and `main` are not modified. Keep this as a Draft PR pending CI and eventual real voicebank acceptance.