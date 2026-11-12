# PhonoWeave v2.4 — existing sound-unit evidence review

## Why this belongs in the product

The product is meant to identify **which singer-specific realizations are worth retaining or recording**, not to continuously build new acoustic models. This patch reuses the already-existing \`orthography_native_units.py\` evidence reports and adds a **review-only** layer to the existing recording planner. The resulting research review is separate from the already actionable native coverage/reclist (v20–v23).

- The original speaker/author notation is authoritative; Unicode strings such as \`i\`, \`ir\`, \`i0\` remain opaque, never converted to IPA or arbitrarily renamed.
- Existing labels that are consistently discriminable across previous context, pitch, and style checks are tagged \`existing_label_model_rendered_distinction_consistent\`. This means **keep the original label for further study**, not approve a new sound/phone or recording slot.
- Existing labels whose separability varies with extraction windows or lacks enough comparable samples are tagged \`insufficient_or_window_sensitive_evidence\`. These are **verification questions**, not automatic merges and not recording requests.
- No change to the existing \`recording_plan\`, its output \`supplement_reclist.txt\`, selected WAVs, OTO aliases, or synthesis-inventory compilation.

## Invocation

Use the existing command and optionally attach **one to five** previously generated user-native \`evidence.json\` reports. Two or more distinct reports are needed for a multi-report consistency screen.

\`\`\`bash
phonoweave plan-recording /local/voicebank \
  --native-reclist /local/original_targets.json \
  --unit-evidence /local/earlier/evidence.json \
  --unit-evidence /local/generic/evidence.json \
  --unit-evidence /local/later/evidence.json \
  --max-lines 8 --json
\`\`\`

The report gains a separate \`existing_sound_unit_review\` object with per-label context/pitch/style cross-validation counts, minimum rates over the supplied reports, screening status, limitations and SHA-256 hashes of the input evidence files. If \`--export\` is specified, \`recording_plan.json\` contains this audit; the supplement TXT is **unchanged**.

The acoustic-evidence loader strictly checks known source schema, author labels, DiffSinger model-rendered provenance, numeric candidate-window and classification totals, prediction-match consistency, duplicated trial identities, and contradicting aggregate metrics. Identical copied reports are rejected rather than counted as repeated evidence. Input reports are never modified.

## Real historical LynneQing example

Read-only analysis of three archived v1 evidence files: \`results/earlier/evidence.json\`, \`results/generic/evidence.json\`, \`results/later/evidence.json\` from the preserved user-native acoustic discovery ZIP. Each report covers 42 OTO-relative windows of the same single singer's seven model-rendered pitch/style subbanks; there are three pre-existing author labels \`i\`, \`ir\`, \`i0\`. Comparisons share the same logical bank/context/label trial keys and are NOT independent human singing sessions.

Minimum per-label classification rates across the three window choices:

| Existing author label | Context transfer | Hold out pitch | Hold out rendered style | Review decision |
| --- | ---: | ---: | ---: | --- |
| \`i\` | 14/14 | 14/14 | 14/14 | Existing realization appears distinguishable; retain name, request downstream verification |
| \`ir\` | 14/14 | 14/14 | 14/14 | Existing realization appears distinguishable; retain name, request downstream verification |
| \`i0\` | 13/14 | 12/14 | 10/14 | Window/style sensitive; verify, do not automatically merge or split |

**Note:** Entries are *minimum correct counts across three tests*, not independent attempts, and are not phoneme recognition accuracy. This is known-label supervised classification, not unknown-inventory discovery. The hardcoded 0.85 review fraction is an explicitly *heuristic review threshold*, not a validated perceptual or phone-split acceptance criterion.

The existing v13/v14 retrospective synthesis proxies did not support automatic new context splits. This new audit does not overturn that result. Stable existing labels can be kept without adding any new recording budget.

## Limitations and next practical acceptance gate

- Source v1 reports omit original source PCM checksums and verified phone boundaries. Repeated trial IDs demonstrate only that the reports are *comparable*, not that they come from independent recordings.
- Different report content is not proof of a distinct extraction window; the source v1 schema does not encode an explicit window ID.
- Seven subbanks are **one singer**, not seven speakers; synthetic/renderer shifts may explain discriminability.
- No new unit or alias, no modification to voicebank audio, no approved extra recording line. The output is a small review ledger.
- Before a real new-recording recommendation for a novel sound distinction: verify original continuous WAV/OTO sources, stable independent takes/contexts, then compare actual equal-budget synthesis alternatives.

This step does not need manual labeling and does not generate a singing synthesizer.
