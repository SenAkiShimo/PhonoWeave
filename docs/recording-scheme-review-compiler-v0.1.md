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

You can also select an installed voicebank from the existing local
eight-bank manifest without copying a long installation path:

```bash
python -m phonoweave.entrypoint compile-recording-scheme \
  --manifest ~/.config/phonoweave/voicebanks.json \
  --bank-id bank05 \
  --output ~/Downloads/phonoweave_scheme_bank05_01
```

This manifest is read locally; only a uniquely matched `bank_id` and
existing directory are accepted.

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

## v0.2 source-alias surface role preview

Real bank05 records (5,902 aliases) show distinct pitch-suffixed surface roles:
initial CV (`'- shaA#3`), standalone CV (`shaA#3`), vowel-to-consonant (`a shA#3`), vowel-to-vowel (`ang erA#3`), standalone vowel (`aiA#3`), and modified/special transitions (`a 吸RA#3`). Parsing this grammar is a prerequisite for any engine-specific alias compiler.

The new `alias_roles_preview.csv` adds original alias, source OTO and line, stripped pitch suffix, surface tokens, a conservative role classification, and manual-review status. `summary.json` records `alias_role_counts`. A pitch suffix is stripped only when it agrees with the containing note-named OTO directory. Nonstandard consonant spelling, special modifiers, zero/empty alias, and unknown surface grammar remain explicitly manual-review; no alias spelling, OTO values, bank files or unit decisions are changed.

**Important:** classification of the *existing* alias role does not establish phonetic correspondence, role-scoped realization identity, CV/VV segmentation boundary, or deployable OpenUtau alias mapping. The output remains review-only. Deterministic test fixtures cover core CV, initial CV, VC, VV, vowel, modifier, nonstandard phoneme and mismatched suffix.

## Strict alias reconstruction preview

`alias_compilation_preview.csv` is a **read-only**, round-trip-verified
reconstruction of existing pitch-suffixed CV, initial CV, vowel, VC and VV aliases.
Every record retains its source OTO, line number, existing alias, role, and
candidate unit linkage if applicable. Unsupported role grammar, unmatched pitch
suffix, failed round trips and global alias-key collisions are individually
flagged and withheld. A collision is checked across all source OTO files,
including neighboring `EX` directories.

`summary.json.alias_compilation_preview` reports permitted reconstructions,
abstentions and collisions. Existing source aliases are never renamed and
no production `oto.ini`, phonemizer dictionary, new timing or synthesis
assignment is exported. A round-trip pass is a structural check, **not**
an engine-validity or acoustic-effect validation. In particular, a VC/VV
transition does not imply a safe mapping to an initial-CV synthesis inventory
unit. A later convention adapter must define engine requirements separately.

## Pitch-scoped recording coverage and review queue

The review package now includes `recording_coverage_matrix.csv` and
`recording_review_queue.csv`. Each inventory unit is inspected by observed
pitch group and existing initial-CV/standalone-CV alias evidence. The matrix
reports unique alias references, source WAV references, observed syllables,
and potential cross-pitch discrepancies. The queue prioritizes discrepancies
and ambiguous context assignment ahead of already observed units.

**Important limits:** source-WAV references are not distinct acoustic
replications. The matrix must not interpret `mapping_scope_unresolved` or
`not_traceably_mapped` as unrecorded syllables. Internal-CV decisions remain
unassigned until a validated role-aware convention adapter exists. These
outputs are coverage audits and human-review priorities, **not** recording
orders or executable OpenUtau alias installations.

## Internal CV sequence alignment (candidate evidence)

`internal_context_links.csv` provides source-traceable, **review-only** links
between role-scoped inventory groups and existing VC transitions. The adapter
requires a complete recording filename syllable sequence, one initial anchor,
matching event count, strictly increasing OTO offsets, a matching prefix-vowel
and a parsed subsequent CV whose base consonant agrees with the alias. Unknown
w/y shorthand, variant symbols and mismatched sequences abstain instead of
silently guessing a realization. Its counts appear under
`summary.json.internal_context_links`.

A matched link indicates structural evidence, not a validated acoustic
realization, independent WAV sample, phonemizer assignment or production
alias mapping. No new `oto.ini` is emitted and no supplement order is
automatically issued from these links.

## Internal candidate evidence in coverage decisions

Sequence-aligned internal candidates are now consumed by the existing
`recording_coverage_matrix.csv` and `recording_review_queue.csv` rather than
reported only in a separate file. The new status
`internal_candidate_evidence` replaces `mapping_scope_unresolved` only when
the link provenance, pitch, VC alias role, and unit context all agree.
The queue action becomes `review_internal_candidate_mapping`, never
`retain_existing_aliases` or an automatic recording order. The matrix includes
candidate counts, referenced WAV counts and syllables. Unlinked cells remain
unresolved; no source samples or alias configuration are changed.

## Consolidated operational review plan

`recording_action_plan.md` is the main human-facing deliverable.
`recording_action_plan.json` preserves the same decisions and source
references for downstream interfaces. They group repeated per-pitch review
queue items by review action and inventory unit while retaining all affected
pitch subbanks, selected syllables and relevant original OTO/WAV locations.
The document orders concrete alias and internal-context checks before
provisional split/merge decisions. Existing supplemental diagnostic lines
are separate **optional candidates**, not automatically required recordings.
It does not install aliases, infer missing independent takes or alter source
OTO/WAV files. Once reviewer evidence is available, a later policy compiler
can convert accepted decisions into a production recording specification.

## Human decision intake (read-only; experimental)

The `review-recording-scheme` command now creates an explicit approval
template tied to the exact action-plan SHA-256. All pitch-specific tasks
(default `pending`) and all proposed diagnostics (default `defer`) remain
listed, even though the human-facing plan groups repeated pitch tasks.

```bash
phonoweave review-recording-scheme template \
  --plan ~/Downloads/phonoweave_scheme_bank05_09/recording_action_plan.json \
  --output ~/Downloads/phonoweave_decisions_bank05_01.json
```

After actual human source review, set `reviewer_id`, individual
`task_reviews[].resolution`, `note` and `evidence_reference`.
Permitted resolutions depend on the check type. Every diagnostic approval
also requires a rationale and a `related_task_id` with recorded same-base
review evidence. Unreviewed proposals cannot be approved. Do **not** edit
`source_plan_sha256` or task identity fields.

```bash
phonoweave review-recording-scheme compile \
  --plan ~/Downloads/phonoweave_scheme_bank05_09/recording_action_plan.json \
  --decisions ~/Downloads/phonoweave_decisions_bank05_01.json \
  --output ~/Downloads/phonoweave_reviewed_bank05_01
```

The output includes `reviewed_scheme.json` and an
`approved_diagnostic_reclist.txt` containing only explicitly approved
pre-existing candidates. An empty reclist is expected when no reviewer has
approved any candidates. The result remains **diagnostic-only** and does not
write a production UTAU/OpenUtau alias dictionary, oto.ini or audio files.
A regenerated plan invalidates old decision templates to avoid stale approvals.

## Canonical alias export and source-evidence precheck

Each new compilation exports `alias_records.jsonl` containing **lossless**
source aliases, source WAV references, lexical roles, preview compilation status,
and numeric OTO metadata. Presentation CSV fields may prefix spreadsheet-formula
characters with an apostrophe. Never use these CSV values as production alias
strings; machine consumers must use canonical JSONL.

```bash
python -m phonoweave.entrypoint precheck-recording-scheme \
  --plan ~/Downloads/phonoweave_scheme_bank05_11/recording_action_plan.json \
  --alias-records ~/Downloads/phonoweave_scheme_bank05_11/alias_records.jsonl \
  --output ~/Downloads/phonoweave_bank05_precheck_01.json
```

This first precheck reports exact source-alias evidence for cross-pitch
review tasks (e.g. `ceng` across A#3/D#4/G#4), retaining raw strings
and OTO/WAV provenance. It cannot prove acoustic presence, approve a
recording, or change a human review decision. Other review categories
remain pending while more conservative rules are developed.
