# PhonoWeave v3.1 — current voicebank WAV/OTO routing, no historical ZIP required

## What changes for actual voicebank users

The existing **Recording Plan / 补录方案** page's asset viewer now defaults to **当前声库 WAV / OTO**. Select the voicebank directory in the existing field and press **查看已有素材**. The server scans the current local `oto.ini` and WAV files read-only, creating an alias → source WAV/decoded PCM ledger. The old historical-index ZIP workflow remains available under **历史来源索引 ZIP**.

The page shows filters by subbank, original label, following context and review status, plus CSV/JSON downloads. It does not create a new recording scheme, invent phonemes or alter the original supplemental recording plan.

## Conservative identifier rules

- **Without an explicit author's mapping:** display full original OTO aliases and linked WAV paths. No guessing language, pitch prefixes, IPA, phone class, or CV segmentation. Original label is marked unresolved.
- **With existing author target JSON and bank_alias_prefixes:** optionally parse explicit original labels (`i`, `ir`, `i0`) and following contexts, with exact declared prefix matching and a required separator after the label. Bare `i` must not accidentally match `ir`.
- A single long WAV holding multiple OTO aliases is one existing decoded PCM source, not many independent singing takes.
- Exact decoded PCM copies stored at different paths are reported as duplicate-path groups; they are not independent takes.
- Missing or unreadable referenced WAVs, OTO entries pointing to a different subbank, duplicate alias options and local WAVs without OTO entries are visible as source/configuration review, not automatic requests to rerecord.
- A WAV reference resolving outside the voicebank is **not read**, and the external path is not exposed by this ledger.
- This is a file/OTO provenance inventory, not verification of OTO boundaries, phonetic equivalence or real vocal-performance independence.

## Outputs

The read-only scan uses `phonoweave.native_live_assets_v31.scan_live_voicebank` and returns schema `phonoweave.live_voicebank_asset_routing.v3.1`. The GUI asset-summary and CSV/JSON download endpoints accept either historical v2.9 ledgers or current v3.1 results. `source_status` in CSV distinguishes local readable, missing, damaged, cross-bank and unreferenced source options.

Historical source inspection remains separate from live mode, clearly labeled to avoid confusing old evidence with current audio. GUI state and downloads for the asset viewer remain separate from the original `recording_plan` object, `supplement_reclist.txt` and existing review lists.

## Test scope and scientific limits

Small generated PCM WAV/OTO test voicebanks cover long-WAV multi-alias storage, exact PCM duplicates, author-declared bank prefixes, unreferenced or broken WAVs, missing or out-of-bank paths, zero-OTO banks, browser preview/download and stale-result invalidation. Full CI also includes prior Mandarin and user-native recording-plan regressions.

Live analysis operates on the voicebank folder selected at that moment, but does not measure savings in singing minutes, decide which acoustically similar units to merge, or automatically infer unknown phonemes. It is one necessary operational component toward PhonoWeave's real goal: finding useful, minimal speaker-specific recording sets.

The original voicebank is never modified. Keep the research PR draft until CI finishes and the live scanner has been checked against an actual full voicebank with original WAV and OTO files.