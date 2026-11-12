# PhonoWeave v2.6 — first author inspection checkpoint

## Practical objective

This is the first limited **author-acceptance checkpoint**, NOT a new recording assignment. PhonoWeave is for singer-specific phonetic realization and economical voicebank design, not an end-to-end singing synthesizer or a separate phonetic annotation platform.

The v2.5 strict source-disjoint, matched-material feasibility screen could not evaluate a merge for existing \`i\`, \`ir\`, \`i0\` with the old seven-bank source index. The correct next step is **not** to recommend recording 294 environments. Instead generate **up to 3 understandable, source-traceable author review examples** for this known set of existing labels.

## Output

\`python -m phonoweave.native_validation_preview_v26 --source-archive OLD_v2.zip --label i --label ir --label i0 --max-cases 3 --output NEW.json\`

Each case includes the original user labels, full original OTO alias strings, original long WAV filename, bank, following context and source PCM identity. The algorithm only selects an example if *all supplied labels* occur in the **same PCM source WAV** in the **same bank and following context** and their labels/aliases are not ambiguous. It prefers distinct contexts, original WAV filenames, banks and styles rather than arbitrarily outputting three aliases from the same long audio. No acoustic clustering or synthesis algorithm is trained.

It is still **one historical physical source** per case, not independent repeated takes. This is a human-comprehensibility sample, not a statistically powered test or a new reclist. No labels, recording rows or audio are generated.

## First author checks (the user does **not** need to record anything now)

1. Are these original labels/aliases correctly referring to the intended separate realizations?
2. Are the distinctions actually valuable to preserve in the user's singing-synthesis setup?
3. Would those contexts be straightforward to record **if** a future independent source comparison were needed?

A failed check is useful: annotate the mapping or change the candidate; never auto-create a missing phoneme or recording line.

## Scientific and workflow limits

- Existing source ZIP has filtered OTO-relative feature rows from one model-rendered voicebank; no independent human take identity or validated phone boundaries.
- The three selected cases are not a balanced linguistic test or a verified optimized reclist, just a transparent first acceptance checkpoint.
- Source PCM fingerprints are provenance links, not proof of independent sessions.
- \`max_cases\` is deliberately at most 8, default 3. Never automatically use this list as a \`supplement_reclist.txt\`; it is a separate **inspection preview**.

The source files and voicebank remain unchanged. This branch is intentionally isolated from \`main\`.
