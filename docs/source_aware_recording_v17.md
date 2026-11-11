# PhonoWeave v17 — Source-aware, bounded diagnostic recording planner

## Why this change exists

PhonoWeave helps singers **analyze the realizations in their own recordings, inspect coverage, and generate economical recording schemes**. It is not intended to become another renderer or a universal phone classifier. The v16 bounded Mandarin supplemental-reclist tool prioritizes OTO entry counts. Multiple OTO entries may point to one physical WAV, and identical WAVs may be copied to another subbank, so that count alone can misrepresent available recording sources.

This v17 feature reuses the existing legal Mandarin candidate generator and the v16 budgeted scheduler, but adds **audio-source-aware ranking**: rank legal candidates by the number of distinct decoded PCM waveforms already present, then by the number of OTO directories containing audio and distinct WAV paths, then existing OTO entries and syllable spelling. Thus ten aliases from one WAV do not look like ten independent recordings. Byte-identical decoded audio copied into a different file counts once; no fake new unit identities are created.

It also treats items in a user-supplied reclist as *already scheduled work*, not as recorded evidence. If the old diagnostic plan calls for two lines in a context and one already exists in the user's schedule, the new planner proposes at most one new line, and reports the preplanned line separately. An already scheduled line is not used to claim acoustic evidence or confirmed phone coverage.

## Usage (unchanged command)

```bash
# Previous v16 behavior remains available unchanged:
phonoweave plan-bounded-reclist /path/to/voicebank --max-lines 8

# Source-aware preview: never writes by default
phonoweave plan-bounded-reclist /path/to/voicebank \
    --source-aware --max-lines 8

# Only if you explicitly want new output files
phonoweave plan-bounded-reclist /path/to/voicebank \
    --source-aware --max-lines 8 \
    --existing-reclist /path/to/already_scheduled.txt \
    --write-reclist /new/path/extra_recordings.txt \
    --write-report /new/path/source_rationale.json
```

`--source-aware` only changes the preview/optional output of the bounded supplemental planner. It does not change the prior unbounded `evidence-gaps` workflow, add an alias, modify OTO files, or adjust the production synthesis inventory.

## What the new measurement actually means

The source counter decodes WAV audio with Python's `wave` reader and hashes the exact sample bytes plus channel, sample width and sample rate. It counts *distinct digital PCM content* per recognized `(Mandarin initial, final)` from the existing OTO alias parser. A source WAV with multiple aliases counts once for that pair; bit-for-bit identical digital copies across files count once. Missing or unreadable WAVs do not count as audio evidence and produce a diagnostic count.

This is **not an estimate of independent vocal performances**. Two different PCM hashes can still represent copies of one performance resampled, edited, trimmed, or pitch shifted; two files with identical underlying vocal gestures need not be acoustically equivalent. Nor does a unique WAV prove the recorded realization or boundary is usable. OTO entries are used to identify **which labeled candidate a WAV is associated with**, not to determine true phonetic boundaries or automatic new phone identities. None of the v9–v15 exploratory motif results is promoted into a split/merge authorization.

## Planning rules

1. Reuse `analyze_voicebank_inventory` → `build_evidence_completion_plan` → `build_supplement_plan`. Only `supplemental_recording` gaps may produce new lines, never perceptual-only gaps or speculative v9–v15 motifs.
2. Reuse `_candidates` for legal Mandarin syllables and their requested context family; do not invent IPA or phonemes. Sort within each target context by distinct PCM count (ascending), audio-bearing OTO directory count, distinct WAV path count, OTO count and syllable.
3. Enumerate the original fixed carrier-pattern lines for candidate syllables and replicates. Lines already in the supplied schedule fill *planning slots only*. Fill remaining slots with new lines. Refuse duplicate new lines.
4. Reuse `budget_recording_selection` to prioritize one line per outstanding base/context family, then fill further requests, up to `--max-lines`. Report deferrals and unfilled target contexts. This is not mathematically globally minimal across all possible script/phonetic combinations.
5. No automatic changes to existing recordings, OTO, alias spelling, pitch policy or synthesis-unit identity. A "line" may still require multiple pitch takes.

The new JSON adds `coverage_measure`, `already_scheduled`, `diagnostic_slots_requested`, `existing_distinct_pcm_sources` for selected lines, source read errors, and `no_new_synthesis_units_authorized: true`. The old `--source-aware`-off output and ordering remain unchanged.

## Testing and remaining limitations

New tests include PCM digital copy detection, distinct sample rate/content, missing/broken WAV, inflated alias-to-source counts, ranking reversal, no duplicate scheduled work, one-syllable carrier repetition, budget cap, perceptual-only exclusion, no legal candidates, JSON provenance, CLI anti-overwrite and existing v16 regression.

A full private voicebank with complete OTO and original WAVs is needed to assess actual reduction in recording time and improvements in production coverage; historical v9–v14 OTO-selected 600 ms excerpts are not an adequate substitute. Currently Mandarin-specific because the legal-syllable selector is Mandarin-specific. General language-native planning must use a future user-supplied, validated language candidate source, not hardcoded Mandarin syllables.