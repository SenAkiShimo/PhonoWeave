# Language-neutral literal reclist budget planner (v0.1)

This milestone integrates the previously separate **lossless text reclist importer**
with the main recording-review feature branch, then adds a deterministic
coverage-first budget planner. It is an **engineering baseline**, not a
phonological recording-unit optimizer.

## Usage

```bash
phonoweave inspect-reclist template.txt --separator underscore --json
phonoweave plan-literal-reclist template.txt --separator underscore --budget 60 --output literal_budget.json
# or disable adjacent surface-token pairs
phonoweave plan-literal-reclist template.txt --budget 60 --no-pairs --output token_budget.json
```

The planner interprets **only literal tokens and ordered immediately adjacent
token pairs** in source recording lines. It does not parse syllables into IPA,
detect phoneme boundaries, infer CV/VC semantics, or verify singability. The
original file is never modified; output uses exclusive creation.

The greedy procedure selects the line with largest *new* weighted type coverage
(tokens default weight 1, pairs weight 2), breaking ties by earliest source line.
The budget is a ceiling, not a requirement to select duplicate zero-gain lines.
Coverage metrics are *types*, not source occurrences or acoustic frequencies.
The algorithm is deterministic but does not solve exact weighted set cover.

The JSON retains source line numbers, raw selected text, tokens, newly covered
types per selection, uncovered types and explicit alias/OTO export disables.
No absolute path or recording audio is embedded in the generated report.

## Follow-up

Add language-module phoneme mappings and then compare this pre-registered
surface-coverage baseline with evidence-backed speaker-specific choices at
matched **actual** recording durations. Do not treat higher literal pair coverage
as evidence of resynthesis or perceptual benefit.

## Validation

```bash
pytest -q tests/test_reclist.py tests/test_reclist_budget.py
```
