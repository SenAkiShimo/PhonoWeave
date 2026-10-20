# Alias naming and OTO annotation task compiler (v0.1)

The review pack now supports a **separate opt-in manual task** path from the existing split directives. This is not an OpenUtau alias exporter or a validated split strategy.

## Create a review template

```bash
python -m phonoweave.alias_oto_task_compiler template \
  --pack ~/Downloads/phonoweave_review_pack_01 \
  --output ~/Downloads/phonoweave_alias_reviews_01.json
```

Every source-backed internal VC transition yields an explicit **pending** target. A surface-context cue without original OTO provenance does **not** get a new-alias target. The template contains frozen line, syllable, pitch, OTO line, WAV and original alias identity. The existing review pack hashes are verified, and the template is pinned to those artifacts.

For each reviewed target, set `decision` to `approve_annotation` or `reject`. A proposed annotation must supply a collision-free `new_alias` (conservative ASCII syntax), `intended_role: "internal_cv"`, `role_verified: true`, `boundary_verified: true`, and a substantive `reviewer_note`. Set `reviewer_id` on the template. **Do not approve a task solely from its inferred name or old VC alias.**

## Register an independent source-context review

The approval compiler also requires a matching `reviewed_scheme.json`, created by the existing `review-recording-scheme` workflow from the **same** source action plan. For this internal candidate, the appropriate `(unit_id, pitch_group)` must have `candidate_accepted` with a reviewer and evidence reference. Unreviewed or unrelated units cannot authorize a new annotation task. Both reviews require actual acoustic/source inspection; test data must never be mistaken for authorization.

```bash
python -m phonoweave.alias_oto_task_compiler compile \
  --pack ~/Downloads/phonoweave_review_pack_01 \
  --reviews ~/Downloads/phonoweave_alias_reviews_01.json \
  --reviewed-scheme ~/Downloads/phonoweave_reviewed_scheme_01/reviewed_scheme.json \
  --output ~/Downloads/phonoweave_alias_oto_tasks_01
```

The output includes `alias_oto_annotation_tasks.json`, `ANNOTATOR_OTO_TASKS.csv`, and `TUNER_PENDING_ROUTING.md`. The CSV is spreadsheet-safe presentation data, never a canonical input for alias installation. An empty CSV (no approved work items) is a valid conservative result.

## Safety and scope

- Source WAV, OTO and source alias strings are not modified.
- Existing aliases are reserved; new alias collisions or duplicate proposed names are rejected.
- Approving a **manual OTO measurement task** does not authorize actual installation or create any timing values. All OTO values and phonemizer routes remain `null`.
- Previously generated `recording_review_pack` must match its recorded SHA-256 artifact checksums. Old alias reviews cannot be applied silently to a regenerated pack.
- The old VC transition may not contain a separately usable internal CV realization; it must be audited acoustically before any mapping is adopted.
- Initial-only lexical context assignments and unsupported role classes are intentionally held out.
