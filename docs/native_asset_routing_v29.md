# PhonoWeave v2.9 — original alias / physical WAV routing ledger

## Goal: reduce unnecessary recording work without merging sounds

PhonoWeave's core job is **speaker-specific phonetic realization and efficient recording-list design**. The author has confirmed that \`i\`, \`ir\` and \`i0\` are different existing intended realizations. The v2.5–v2.8 reviews cannot yet justify a phonetic unit merge or a fresh recording demand; therefore v2.9 inventories exactly how existing OTO-relative aliases map to **already recorded PCM source assets**.

It deliberately separates two concepts:
- **Safe existing source-container reuse**: a long WAV may hold many OTO aliases. Reference that existing WAV and preserve each original alias/segment identity.
- **Unverified sound substitution**: the same WAV holding \`i\`, \`ir\` and \`i0\` does **not** prove that their audio segments can substitute for one another. No cross-context or cross-label substitution is authorized.

## Implementation

\`phonoweave.native_asset_routing_v29\` reads the same frozen source-linked v2 feature archive used previously. It creates:
- \`original_alias_routes\`: one per (bank, original author label, following context), with exact original alias(s), WAV path(s), source PCM identity and a flag for multiple-source ambiguity.
- \`existing_source_containers\`: one per (bank, decoded PCM hash), with original long WAV paths and count of separate label+context links held in that container.
- \`same_container_cross_label_examples\`: source-provenance references where all selected labels appear in the same long WAV, **explicitly denying** auto merge/substitution.

The read-only module can be used independently:

\`\`\`bash
python -m phonoweave.native_asset_routing_v29 \
  --source-archive /local/PhonoWeave_User_Native_Batch_Discovery_v2.zip \
  --label i --label ir --label i0 \
  --output /local/new_existing_asset_routes.json
\`\`\`

Or as an optional, non-mutating section within the established recording workflow:

\`\`\`bash
phonoweave plan-recording /local/existing_voicebank \
  --native-reclist /local/original_targets.json \
  --asset-routing-archive /local/PhonoWeave_User_Native_Batch_Discovery_v2.zip \
  --asset-routing-label i --asset-routing-label ir --asset-routing-label i0 \
  --max-lines 8 --json
\`\`\`

When attached, the existing supplement TXT, source-review checklist, author acceptance and recording line budget remain unchanged. Only an \`existing_source_asset_routing\` section is added to the JSON report. There is no new phonetic clustering, resampler, GUI page or recording-task generator.

## Real read-only LynneQing index result

The historical source index has 7 singer subbanks. Restricting to known author-original \`i\`, \`ir\`, \`i0\` gives:

- **917** accepted OTO-linked source windows in total.
- **910** distinct (subbank, original label, following context) routing entries: **130 per subbank**.
- **70** distinct existing decoded PCM source containers across banks: **10 per subbank**.
- **7** routes with two distinct WAV sources, one per bank for \`i\` in following context \`m\` (\`eriin.wav\` and \`vcsection2.wav\`). They are marked **source-choice review**, rather than silently selecting a winner.
- At least **140** bank+context combinations have all three original labels in the same physical source WAV (as already established in v2.6's strict one-alias-per-label preview). This is *co-storage*, not one audio segment or independent repeated vocal takes.

The number **70** is the number of existing digital containers, **not** an optimal new recording-list size, singing-session count, or acoustic minimum. The counts apply only to filtered OTO-derived windows; full original WAV/OTO has not been independently re-read in this workspace. Missing source-index rows do not prove that an original recording was absent.

## Author-review status, limitations, and production decisions

- \`i\`, \`ir\`, \`i0\` remain distinct exactly as the author confirmed.
- Recorded environments \`a\`, \`ei\`, \`hw\` retain **unknown future recordability**. This routing ledger has no mandate to reassess or schedule them.
- Asset links describe observed source files; even two aliases pointing to the same WAV need not be acoustically substitutable.
- Exact PCM hashes are not independent singer-performance IDs. Reused source files or aliases must not be counted as multiple independent takes.
- Source-index ties are flagged, not collapsed. No automatic alias merge/split, unit replacement, new recording assignment, or modification of WAV/OTO or the original voicebank.

### Next product acceptance

A useful next *small* implementation step is to expose existing alias coverage and ambiguous source choices in the current Recording Plan GUI's existing review section (no new GUI system). Beyond that, genuinely reducing new recording *time* requires checking complete original source WAV/OTO plus authored reclist constraints, not reinterpreting these 910 observed alias routes as 910 recording prompts.
