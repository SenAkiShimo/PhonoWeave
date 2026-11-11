# PhonoWeave v16 — bounded practical supplemental recording list

**Purpose:** Make the existing voicebank coverage→evidence-gap→supplemental-recording workflow easier for one singer to use. This is not a new phoneme detector, a synthesis engine, or a claim that existing acoustic motifs imply separate units.

## What changes

The original `evidence_gap`, `supplement_plan`, `diagnostic_selector` and `supplement_reclist` logic is reused without changing its scientific criteria, Chinese syllable legality, or text formatting. This small step:

- Sets an explicit global limit on **new recording-script lines** (default **8**). It does not treat lines as one total recording take; the original repeat-across-pitch-subbank policy still applies.
- Gives each candidate **base-unit/context-family** one chance before allocating additional replicates to that same family. Within a family, keep the original selector's least-observed-first choice; first consider high-priority gaps, then medium/low.
- Excludes exact duplicate proposed lines, and optionally skips lines already scheduled in a separate user-supplied reclist. Skipping an existing line is only de-duplication of work, **not proof that the sound is acoustically covered**.
- Keeps reasons, original OTO observation counts, priorities, omitted-by-budget entries, and the old selector's unfillable targets in the report. Only gaps marked `supplemental_recording` are eligible; `perceptual_validation` gaps do not generate recording tasks.
- Never accepts v9–v15 acoustic motif IDs as new phones, changes aliases, touches OTO files, edits source audio, or automatically approves splits. No arbitrary-language coverage is claimed: existing legal-syllable selection is **Mandarin-specific**. User-native phoneme spellings and other languages need their own validated candidate generator before this planning layer can generalize.

## Run

From an existing PhonoWeave checkout on the v16 branch:

```bash
# Preview eight suggestions without writing anything
phonoweave plan-bounded-reclist /path/to/existing-voicebank --max-lines 8

# Avoid exact lines already in an independently supplied list, write new outputs
phonoweave plan-bounded-reclist /path/to/existing-voicebank \
  --max-lines 8 \
  --existing-reclist /path/to/current_reclist.txt \
  --write-reclist /new/location/new_supplement_reclist.txt \
  --write-report /new/location/recording_rationale.json
```

`--max-lines 0` generates an empty suggestion list and full deferral log. `--json` prints a machine-readable report. Output paths must not exist; default is preview only. The original unbounded `phonoweave evidence-gaps` workflow remains unchanged.

## When it should say no

- A gap requires perceptual evidence but no additional recording: do not add a recording line.
- All candidate lines already appear in the supplied schedule: do not record them a second time; report omission.
- Budget is exhausted: retain the remaining suggestions with `global_line_budget_reached`, not silently discard them.
- The original legal Mandarin selector cannot fill a requested context: retain its `unfilled` record; do not invent unpronounceable syllables.
- Existing v13/v14 experimental seam and continuation evidence has **not** demonstrated a reason to split new synthesis units. This tool must not override v15's non-deployment conclusion.

## Limits

This release is a **small scheduling layer on top of the existing Mandarin voicebank analyzer**, not a new inference algorithm. OTO observation counts are proxies for the currently documented inventory, not sufficient evidence that two recordings are identical or that an extra line will improve synthesis. It does not optimize total minutes, vocal fatigue, session count, or frequency of repeated pitch captures. The user can always modify the resulting text before recording.