# PhonoWeave v15 — research evidence to recording-plan safety gate

## Purpose

PhonoWeave is designed to analyze a speaker's context-dependent phonetic realizations and help decide which synthesis/recording units are useful. It is **not** a singing synthesizer. The earlier v9–v14 audio experiments should not independently produce more recording units.

This small, read-only gate takes the **existing v13 and v14 research summaries**, and answers just one question: **does this particular experimental strategy justify creating new context-dependent recording units?**

The existing \`evidence_gap.py\`, \`supplement_plan.py\`, \`diagnostic_selector.py\`, and \`supplement_reclist.py\` remain unchanged. Other real recording-coverage deficiencies can still be analyzed independently.

## Input and safeguards

- Requires the exact frozen v13 and v14 JSON schemas.
- Verifies shared historical audio-archive SHA-256, matching eligible clip count, source-family split counts, audit-group and observation counts, and v14 candidate budget.
- Requires upstream declarations about train-only fitting, no right-hand target leakage into v14 donor selection, no validated phone labels or actual singing-quality outcomes, and prior retrospective examination.
- Recalculates each metric's reported relative improvement using its paired *lower-is-better* baseline. Reports v13 and v14 metrics separately because **these are distinct experiments with different endpoints and budget settings**.
- Never combines acoustic proxies into a subjective quality score, invents IPA/aliases, changes voicebanks, edits OTO files, creates a reclist, or changes the production synthesis-inventory compiler.
- Fails closed when provenance is missing or contradictory. Writes one SHA-256-bound JSON report at a new path, refusing overwrite.

Matching partition *counts* do not prove identical per-record assignments, and these older held-out groups have been examined during development. These are **retrospective exploratory results**, not untouched confirmation.

## Run

\`\`\`bash
python -m phonoweave.native_recording_gate_v15 \
  --v13-summary /local/v13/summary.json \
  --v14-summary /local/v14/summary.json \
  --output /local/new_evidence_gate.json
\`\`\`

## Actual LynneQing result

Across the historical OTO-curated short-audio archive, the retrospective audit involved 4 carrier families and 28 observations. In v13, context-informed spectral donor selection worsened the held-out spectral continuation proxy by ~32.42%. In v14, seam-aware waveform selection improved its own seam-cost metric ~16.25%, yet worsened its independent continuation-trajectory proxy by ~22.44% and max sample-step transient proxy by ~25.63%.

The resulting status is \`current_context_split_not_supported\`. The emitted recording decision is \`new_units_authorized=false\`, with empty alias and new reclist arrays, and \`existing_voicebank=unchanged\`.

**Scope:** only the tested historical acoustic motifs and candidate strategy. This does **not** assert that all coarticulation splits are useless, that no recording gap exists, or that hearing-quality differences are impossible. It does not globally block the established coverage-based supplemental-recording workflow.

## Tests and limits

13 new synthetic/provenance tests exercise negative outcomes, even hypothetical positive-proxy nondeployment, conflicting sources, group counts, metric tampering, nonfinite metrics, leakage, and refusal to overwrite. Combined local v9–v15 test suite: **102 passed** on Python 3.13. GitHub CI must be checked separately.

No direct production integration, auto-inventory split/merge, independent phone-boundary reference, multi-speaker proof, or perceptual A/B evaluation is claimed.

**Next sensible development:** return to the *existing* recording-coverage gap selector and improve its practical recording recommendations where actual coverage is insufficient, rather than inventing another acoustic model.
