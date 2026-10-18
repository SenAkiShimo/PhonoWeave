# Batch alias naming review (v0.1)

The batch layer groups existing, source-backed internal VC candidates by
`(unit_id, context)` across observed pitch layers. It replaces repetitive
*naming-policy entry*, not individual acoustic or annotation review. All
outputs are review-only; no voicebank WAV, `oto.ini`, production alias map,
phonemizer setting or OTO timing is written.

## Generate a batch template

```bash
python -m phonoweave.alias_batch_review template \
  --pack ~/Downloads/phonoweave_review_pack_01 \
  --output ~/Downloads/phonoweave_alias_batch_01.json
```

The JSON includes frozen source/plan fingerprints, groups with `target_task_ids`,
and one per-source target per internal candidate. Every group defaults to
`decision: "pending"`; every target independently defaults to `pending`,
`role_verified: false`, `boundary_verified: false`.

To review a **naming policy**, first inspect the group's existing source
provenance and convention. Fill in `reviewer_id`, then, for an accepted group:

```json
{
  "decision": "approve_naming",
  "alias_format": "{unit_id}_{pitch_group}_{task_id}",
  "reviewer_note": "Naming rule checked against source alias convention"
}
```

This is an example only; the rule is not approved by default. Supported field
names are `unit_id`, `context`, `pitch_group`, `task_id`, `line_id`,
`syllable`, `syllable_index`. Formatting operations, dotted attribute access,
indexing, and unknown fields are rejected. Rendered aliases must satisfy the
existing conservative ASCII alias grammar. `task_id` provides a distinct name
for each specific source; schemes that render the same name for different
source targets are rejected, as are collisions with **any original alias**.

## Preview grouped naming proposals

```bash
python -m phonoweave.alias_batch_review preview \
  --pack ~/Downloads/phonoweave_review_pack_01 \
  --batch-reviews ~/Downloads/phonoweave_alias_batch_01.json \
  --output ~/Downloads/phonoweave_alias_batch_preview_01.json
```

This reports proposed names and OTO source references for approved **naming
rules**, even while all per-source annotation decisions remain pending. A
preview is never an alias installation or an approved OTO work order.

## Independent per-target decisions and compilation

**Do not batch-approve acoustic boundaries.** Only after inspecting each
original WAV, OTO row, phonetic role and boundaries may a specific target have:

```json
{
  "decision": "approve_annotation",
  "role_verified": true,
  "boundary_verified": true,
  "reviewer_note": "Source boundary and internal-CV role checked manually"
}
```

This additionally requires an independently accepted `(unit_id, pitch_group)`
source-context review in `reviewed_scheme.json` from the **same frozen action
plan** (`review-recording-scheme` workflow). Group naming approval alone
cannot make any annotation task. Use:

```bash
python -m phonoweave.alias_batch_review compile \
  --pack ~/Downloads/phonoweave_review_pack_01 \
  --batch-reviews ~/Downloads/phonoweave_alias_batch_01.json \
  --reviewed-scheme ~/Downloads/phonoweave_reviewed_scheme_01/reviewed_scheme.json \
  --output ~/Downloads/phonoweave_alias_batch_tasks_01
```

The existing strict `alias_oto_task_compiler.compile_oto_tasks` remains the
final validation gate. Output consists of `alias_oto_annotation_tasks.json`,
`ANNOTATOR_OTO_TASKS.csv`, and `TUNER_PENDING_ROUTING.md`. **Even a genuinely
approved annotation task requests manual OTO measurement only.** Its OTO
numbers and phonemizer route remain null; the tuner guide never claims a
usable context routing table. An empty approved task list is valid.

Templates are pinned to the exact source review pack, action plan and target
identities. A regenerated pack, missing/duplicated target, tampered group
membership, or unsafe/colliding name fails closed. Original source files
are never modified.
