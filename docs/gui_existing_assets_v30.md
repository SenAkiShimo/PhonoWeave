# PhonoWeave v3.0 — historical existing-WAV routing in Recording Plan GUI

## Purpose

Make v2.9's source-provenance inventory directly useful in the **existing** Recording Plan UI. No new acoustic algorithms, phone recognition models, new standalone application, proposed extra recording lines, or voicebank mutation.

The same page now has an **optional, clearly labeled historical archive area** independent from the active voicebank recording-gap preview. A user supplies their own previously created source-index ZIP (for example the frozen v2 research index) and 2–5 existing original labels such as \`i, ir, i0\`. The server reuses \`native_asset_routing_v29.run_archive\` to display:

- (subbank, original label, following context) -> original OTO alias(es) -> source WAV option(s)
- previously observed PCM source-container counts and flagged multiple-source routes
- filters for subbank, original label, following context and ambiguous mappings
- full download as \`existing_asset_routes.csv\` or \`existing_asset_routes.json\`

The GUI initially renders **no more than 80 filtered routes** for responsiveness; full CSV/JSON contains all routes. CSV cells beginning with spreadsheet formula characters are escaped. Client-side rendering uses \`textContent\`, never HTML injection of untrusted alias strings.

## Read-only boundaries

The historical source-index ZIP can be an older, filtered analysis and is **not a live re-read of a local voicebank's WAV/OTO**. The interface prominently says this. The two GUI previews keep separate state: a new asset preview cannot change the existing \`recording_plan\`, supplement TXT, scheduled lines, or OTO/source verification tasks.

A failed asset preview invalidates prior asset downloads. Changing archive or labels hides the old client preview; in-flight responses are guarded by a generation counter so an older request cannot populate a newly edited form.

All source reuse is **long-WAV file-container reuse**. It never implies that \`i\`, \`ir\`, \`i0\` are interchangeable audio segments, phonemes that should merge, or an optimized recording-time minimum. The previously accepted author decisions stay intact: keep the three original realizations distinct and do not treat \`a/ei/hw\` as approved independent recording tasks.

## Validation

Synthetic ledger and live loopback HTTP tests cover:
- page endpoints, preview/download availability, independent recording-plan state, previous result invalidation, cross-origin request protection and invalid labels
- CSV source rows, correct original aliases and safe handling of formula-like strings
- no new phoneme/alias/recording changes and bounded, text-only UI rendering

The full original LynneQing WAV/OTO corpus is unavailable in this workflow; this GUI integration does **not** claim new acoustic validation or measured saved recording minutes. Preserve the work as a Draft PR until CI passes. Neither original voicebank nor \`main\` is modified.
