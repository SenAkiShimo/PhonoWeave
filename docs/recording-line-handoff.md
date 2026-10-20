# Line-aligned annotation and tuning handoff

`python -m phonoweave.recording_line_handoff` compiles a source-backed,
non-destructive *line-by-line* plan from a selected speaker-context budget.
It complements, rather than replaces, the unit-level split/OTO and tuner guides.

```bash
python -m phonoweave.recording_line_handoff \
  --blueprint /path/to/reference_reclist_blueprint.json \
  --budget /path/to/speaker_context_budget_plan.json \
  --units /path/to/recording_units_review.csv \
  --internal-links /path/to/internal_context_links.csv \
  --output /path/to/new_line_handoff_directory
```

## Outputs

- `recording_line_handoff.json`: lossless line, pitch, syllable-position,
  original WAV/OTO reference, unit/context candidate and review-status records.
- `ANNOTATOR_LINE_TARGETS.csv`: spreadsheet-safe per-syllable checklist.
  **Do not** copy its escaped alias strings into a production configuration.
- `ANNOTATOR_LINE_GUIDE.md`: line-specific source and boundary review notes.
- `TUNER_LINE_GUIDE.md`: line-specific context and listening review notes.

## Evidence restrictions

The compiler accepts internal-context assignments only where an explicit
sequence/OTO candidate link matches the chosen WAV, OTO file, pitch, syllable
index, preceding syllable and known inventory unit/context. It never infers
unrecorded sounds from the absence of an alias, or derives OTO timing from
spelling. Protected non-internal CV units remain separate: they are not
incorrectly reported as internal-candidate gaps. The generated files are
**review-only**, never phonemizer routing tables or new OTO parameters.

Every new plan must be independently reviewed against the actual recordings.
Original source recordings and annotations are never changed by this command.
