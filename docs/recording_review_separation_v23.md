# PhonoWeave v2.3 — make the recording plan actionable

## Scope

Keep the same **user-native template** workflow and the existing Mandarin
workflow. No new sound discovery model, IPA mapping, phonetic segmentation or
automatic editing of WAV / OTO / alias files.

The v2.2 native planner already distinguishes exact source matches, ambiguous
long-WAV aliases and genuinely missing files. v2.3 turns the distinctions into
**separate action queues**, not one confusing "please rerecord" list.

## Native plan outputs

1. **Record (TXT \`supplement_reclist.txt\`)** — only templates whose bank
   status is \`missing_audio\`, within the author's requested line budget.
   A missing file in an otherwise populated long-WAV bank is *not* enough
   evidence to assert a missing vocal realization: first check the mapping.
2. **Fix OTO registration (\`oto_configuration\`)** — readable audio is
   present in the requested bank, but the author-defined alias has not been
   registered for that WAV. Do not rerecord.
3. **Verify audio/source (\`source_verification\`)** — invalid/damaged WAV,
   ambiguous OTO alias-to-WAV matches, missing/unknown mappings, or
   cross-bank references. Check originals/backups and the alias mapping before
   deciding whether recording is necessary.

Each review row preserves the author-supplied \`line\`, \`bank\`, machine
\`status\`, and explanatory \`reason\`. The report retains raw per-bank statuses
and the existing \`oto_review_without_rerecording\` compatibility field.

### Export and GUI

- \`phonoweave plan-recording ... --native-reclist ... --export <new_directory>\`
  outputs \`supplement_reclist.txt\`, \`recording_review.txt\` and
  \`recording_plan.json\`. Existing export directories and voicebank files
  are never overwritten. Standard Mandarin exports keep the old two-file
  behavior when there are no native review tasks.
- The existing **Recording Plan** page now shows separate review tables for
  OTO configuration and source/audio verification and has a review TXT
  download link.
- \`--max-lines 0\` suppresses new recording lines, not review tasks.
  A missing recommendation does **not** mean complete phonetic coverage.

## Example

In a tiny two-subbank synthetic fixture:
- D4/ŋa already has a valid WAV and OTO: **no action**
- D4/坏音 has a corrupted WAV: **verify source**, not rerecord automatically
- D4/待登记 has a readable WAV without matching OTO alias: **fix OTO**
- D4/未映射 has other long-WAV material but no proven target mapping:
  **verify source / alias mapping**
- E4/确实未录 has no WAV: **one supplementary recording line**

The tests use synthetic PCM/OTO files and validate that the export and GUI do
not confuse these cases. This is an engineering guardrail, not a measured
real-world precision/recall rate, verified phoneme inventory or independent
performance audit.

## Deliberate limit

The original LynneQing voicebank WAVs/full OTO are still unavailable in
this workspace. Historical v2 source-index counts cannot prove the final
correctness of a new reclist. Keep the work in Draft until full CI passes
and a full-source audit is possible.
