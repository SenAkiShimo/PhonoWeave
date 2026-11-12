# PhonoWeave v3.3 — speaker-dependent context candidate screening

## Product objective

For the **current user's actual voicebank**, compare whether an existing *base* sound's realized spectrum changes with its following environment enough that a context-conditioned prototype can predict separate recorded source material better than a shared prototype. Preserve the existing author's recording labels. This is a **candidate-retrieval and evidence-prioritization** step for eventual singer-/accent-adaptive split/share decisions, not a universal Mandarin table or an automatic new-phone detector.

This addresses a different question from verifying already-conventional \`i/ir/i0\` labels: **within one base**, do different *following contexts* have stable acoustic differences for THIS singer? All attested base/context pairs are enumerated, rather than picking one already split pair and training a classifier to recognize it.

### Run against the real voicebank

\`\`\`bash
python -m phonoweave.accent_context_comparison_v33 \
  "/Users/yang/Library/OpenUtau/Singers/LynneQing_CVVChn_Mang" \
  --output ~/Downloads/phonoweave_accent_screen_01
\`\`\`

Optional \`--base sh\` restricts which base symbols to show, **but it is not the default**. Without an author map, the existing Mandarin parser proposes phonotactic comparison families such as plain/rounded. These are *linguistic hints for matching existing OTO observations*, **NOT inferred accents, automatically discovered new phonemes or a fixed number of split classes**. The empirical spectrum may favor shared or distinct context-conditioned prototypes; neither possibility is automatically accepted.

For an arbitrary language/notation, provide **exact original OTO alias identities**:

\`\`\`bash
python -m phonoweave.accent_context_comparison_v33 \
  /path/to/voicebank \
  --author-context-map /outside/author_contexts.json \
  --output /outside/new_accent_screen
\`\`\`

Example \`author_contexts.json\`:

\`\`\`json
{
  "schema": "phonoweave.author_context_map.v1",
  "cases": [
    {"bank": "D4", "original_alias": "sh-a", "base_unit": "sh", "context": "plain", "stimulus": "sha"},
    {"bank": "D4", "original_alias": "sh-u", "base_unit": "sh", "context": "rounded", "stimulus": "shu"}
  ]
}
\`\`\`

The aliases must match the author's original OTO **exactly**; examples are illustrative and should be replaced with actual names. No romanization, IPA normalization or translation is performed on author-provided labels. Input context grouping is a *proposed explanatory variable* and may reflect coarticulation rather than an independent synthesis-unit identity.

### Outputs

- \`context_candidates.json\`: ranked base/context pairs, each subbank's source-disjoint status and exploratory MSE comparison, and a few original WAV/alias examples for source inspection
- \`context_review_sheet.csv\`: readable shortlist for humans to inspect actual pronunciation and plan a subsequent renderer comparison
- Each candidate is **UNRESOLVED** and authorizes **zero** new aliases, phone categories, OTO edits or recordings
- Output directory must be new and outside the actual voicebank. Generated documents may reveal local paths and author sound names; keep private

### Comparison design

1. Read the **original** \`oto.ini\` and the corresponding current WAVs. Parse input symbols without fixed IPA assumptions; store original alias, pitch/style subbank, source WAV and exact decoded PCM SHA-256.
2. For each provisional base/context observation, collect 3 short Hann-windowed 12-band log-spectrum snapshots from inside the OTO offset→preutterance range. This is **not an independently reviewed phonetic boundary**. Treat these measurements as **phonetic-acoustic context features only**. Very short, silent or undecodable crops are excluded.
3. Collapse repeated OTO rows from the **same decoded PCM and bank/base/context** into one source-level observation. Do not count multiple slices of the same long recording as independent takes.
4. For every attested context pair within a base, compare identical-source-budget pooled vs separate context prototypes, leaving each tested PCM source out of **both** models. The pooled prototype averages context means equally, rather than allowing an overrepresented context to win by sheer alias count.
5. Report a bank only when both contexts have **at least 2 source-independent heldout opportunities**, with **at least 2 other PCM source groups per context in each fold**. If the available source structure does not support this, **abstain** (no invented metric).
6. \`mean_shared_minus_separate_mse > 0\` means **context-conditioned acoustic prototype predicts the heldout spectrum more closely**, not that a human distinct phone, split alias, or synthesis gain has been established. Negative means the shared prototype was closer on this proxy; it does **not** authorize merging two existing author-defined units. The same singer across pitch banks is still one singer, not independent vocalists.

### Known scientific constraints

This v3.3 does **not** solve unrestricted acoustically blind unit discovery or infer a new category count. The initial Mandarin adapter can only surface contexts represented in the source OTO structure. The short OTO region may contain strong changes due to neighboring vowels or bad alignment, not a useful consonant split. Real source-WAV independence is not assured merely by nonidentical decoded PCM. Some voicebanks will legitimately output only \`insufficient_independent_PCM_groups\`; that does not mean the accent has no distinctive contexts.

**Next acceptance experiment:** inspect source-linked candidate intervals, explicitly validate their phonetic role, then build true *matched-time-budget* audio synthesis A/B with both shared and conditioned donor inventories, holding whole source recordings out. Finally evaluate listener-rated continuity and reconstruction of real heldout singing. **Only after that** may a candidate enter a production recording scheme as an approved split or shared unit.

In particular, a found context pair with different spectrum may simply be normal coarticulation. We must ask whether the split benefits *this singer's synthesis* and yields enough improvement to justify its extra recording cost.

### Test

\`\`\`bash
python -m pytest -q tests/test_accent_context_comparison_v33.py
\`\`\`

Synthetic tests check independent source counts, same-PCM collapse, no forced comparison between author-distinct base labels, exact-map parsing, CSV safety, read-only output, real PCM I/O and source-isolated prediction. Passing engineering tests is not a real-singer acoustic or listening validation.
