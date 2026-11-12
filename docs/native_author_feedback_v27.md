# PhonoWeave v2.7 — author feedback accepted; recordability still unknown

## Accepted result of the first v2.6 author checkpoint

These are **author confirmations**, not inferred phone facts:

- Original OTO aliases for \`i\`, \`ir\`, \`i0\`: **yes**, mapping is as intended.
- Distinct realization IDs \`i\`, \`ir\`, \`i0\` should be preserved for the synthesis design: **yes**.
- Whether the particular following contexts \`a\`, \`ei\`, \`hw\` are feasible for a future genuinely independent recording test: **uncertain**. Keep all three as *unknown*, never interpret uncertainty as consent, rejection, or an instruction to record.

This patch stores these three answers in an explicit source-fingerprint-bound feedback document. It **does not change** the full voicebank or any original aliases.

## Run and data contract

The portable feedback JSON is \`phonoweave.native_author_feedback.v2.7\`, including the exact SHA-256 of the original v2.6 review JSON, two global confirmation fields, and per-reviewed-case recordability. It is validated against the original case IDs and aliases.

\`\`\`bash
python -m phonoweave.native_author_feedback_v27 \
  --preview original_review_cases.json \
  --feedback author_feedback.json \
  --source-archive PhonoWeave_User_Native_Batch_Discovery_v2.zip \
  --max-alternatives 3 \
  --output NEW_feedback_result.json
\`\`\`

The archive is the *same frozen archive* used for v2.6 and is checked by SHA-256. The tool then retrieves up to 3 more source-provenance-linked examples for future technical examination, **excluding the previously reviewed \`a\`, \`ei\`, \`hw\` contexts**. These alternatives are explicitly \`unreviewed\`; they are not marked more recordable or phonetically correct.

The resulting report has \`recording_readiness=blocked_by_unconfirmed_context_recordability\` and three lists that must remain empty: \`automatic_recording_requests\`, \`automatic_unit_merges\` and \`automatic_unit_splits\`. The optional \`attach_author_feedback\` function may add this as a separate review section to an existing recording report, leaving original \`recording_plan\` untouched.

## Real historical preview

On the previous one-singer, seven-subbank source-index archive there are **119** alternative same-PCM candidate cases after excluding \`a\`, \`ei\`, and \`hw\`. The deterministic, three-case diverse preview returns:

| Existing bank | Following context | Historical WAV | Original author aliases |
| --- | --- | --- | --- |
| \`lu_B4\` | \`ai\` | \`vcsection4.wav\` | \`B4i ai\`, \`B4ir ai\`, \`B4i0 ai\` |
| \`mang_D4\` | \`en\` | \`vcsection5.wav\` | \`D4i en\`, \`D4ir en\`, \`D4i0 en\` |
| \`lu_C4\` | \`k\` | \`vcsection15.wav\` | \`C4i k\`, \`C4ir k\`, \`C4i0 k\` |

These are **NOT the next three recording assignments**. Source WAVs are long historical files, and the original OTO aliases may not be straightforward to perform as standalone script lines. The alternative examples are only for future checks using existing material. The author does **not** need to annotate or record anything in v2.7.

## Next narrow development

Preserve the author's confirmed distinctions as candidate unit IDs, but do not infer new phones or unit split cost. Improve the objective comparison's evidence bookkeeping using existing source material; require truly independent recordings and a source-disjoint equal-budget rendered comparison before suggesting optional future verification recording. Do not create a new recording list based on this review.

Source data and private feedback are not uploaded to GitHub.
