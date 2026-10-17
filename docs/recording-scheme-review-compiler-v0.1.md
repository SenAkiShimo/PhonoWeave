# Recording-scheme review compiler v0.1

## Product milestone

Convert an existing voicebank to a **human-reviewable recording-scheme package** using the existing PhonoWeave analysis and intermediate representations. This work connects previously separate core modules; it is not another proof-of-coarticulation experiment.

```text
voicebank with oto.ini/WAV
  -> analyze_voicebank_inventory
  -> build_speaker_profile
  -> build_synthesis_inventory
  -> recording-scheme review compiler
     -> neutral units
     -> observed alias -> candidate unit review
     -> original OTO timing review (unchanged values)
     -> prioritized diagnostic/supplemental items -> reclist
     -> summary and unfilled targets
```

## Usage

```bash
cd ~/Downloads/PhonoWeave
source .venv/bin/activate
python -m pytest -q

# Choose a real, installed voicebank path and a NEW output directory.
python -m phonoweave.entrypoint compile-recording-scheme \
  "/path/to/voicebank" \
  --output ~/Downloads/phonoweave_scheme_review_01
```

If the `phonoweave` shell command is available, the equivalent is
`phonoweave compile-recording-scheme "/path/to/voicebank" --output ...`.

This command never overwrites an existing output folder; it creates a sibling temporary output directory and renames on successful compilation.

## Generated files

| File | Purpose | Operational status |
| --- | --- | --- |
| `synthesis_inventory.yaml` | Existing alias-neutral speaker synthesis inventory | Intermediate representation |
| `recording_units_review.csv` | Contexts, decisions, confidence, evidence, review action | Review only |
| `observed_alias_review.csv` | Source WAV / `oto.ini` location, original alias and candidate neutral unit | **NOT** an installed alias map |
| `oto_timing_review.csv` | Existing timing values with missing-source/unresolved/collision review flags | **No timing changes proposed** |
| `supplemental_targets.csv` | Selected diagnostic syllables, repeats, source coverage and phonetic family | Suggested recording targets |
| `supplemental_reclist.txt` | Human-readable diagnostic recording lines where eligible | Not guaranteed complete reclist |
| `summary.json` | Counts, collisions, withheld assignments, unfilled diagnostics, current limits | Audit |
| `README.txt` | Meaning and safety of outputs | Review only |

CSV fields beginning with spreadsheet-formula prefixes are made safe for manual opening. Source WAV and `oto.ini` files are read-only.

## Design restrictions

- Explicitly preserve `unresolved` decisions; `split_recommended` and `merge_supported` remain **provisional** rather than instructions to delete samples.
- Preserve exact original alias spelling and numeric timing metadata, but **never invent UTAU alias strings or new `oto.ini` boundary values**.
- A candidate neutral unit assignment is made only when the known Mandarin context maps uniquely to one existing inventory unit. For role-scoped contexts, no role is inferred from spelling; require manual review.
- Detect duplicate alias spellings with divergent candidate unit IDs and flag collisions.
- Flag missing source WAVs, unclassified aliases and unsupported inventory groups without filling missing evidence with fabricated acoustic outcomes.
- Compiling a diagnostic reclist does not assert that these are fully valid production recordings. Its selection reuses existing `diagnostic_selector`, `supplement_plan`, and `supplement_reclist` modules. Unfilled targets stay explicit.
- Do not publish absolute local file paths in alias/OTO review sheets. `synthesis_inventory.yaml` preserves the legacy source path field and should be inspected before publication.
- The method does not write an editable `oto.ini`, production alias tables or an automatically optimized final recording list. Those require separate conventions and tested engine-specific export.

## Acceptance criteria

The first milestone is reached when an actual voicebank compiles to the complete set of output files; missing/ambiguous observations are labeled; no original audio/OTO changes occur; and automated tests pass. This validates functioning core output generation, not synthesis quality superiority.

The next milestone is to specify an OpenUtau/UTAU export convention, verify generated aliases against the voicebank's existing scheme, and implement a **preview-only** compiler with collision tests before any automatic editing.
