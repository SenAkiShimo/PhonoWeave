# PhonoWeave v19 — one-page GUI recording-plan integration

## Scope

The existing localhost `phonoweave-gui` home page now offers **Recording Plan / 补录方案**. A new bilingual page at `/recording-plan` calls the **existing v18** `build_recording_workflow`; it does not implement a new scientific classifier.

The page accepts a voicebank directory, maximum new reclist lines (default 8), and optionally a path to an already scheduled reclist. It displays recognized Mandarin observations, diagnostic gaps, lines selected, already scheduled tasks, selected text lines, PCM source counts and warnings.

## Downloads

After a successful preview, the local GUI serves `supplement_reclist.txt` and `recording_plan.json` through browser attachment downloads. The report exists in memory only until another preview or process restart. No voicebank, oto.ini, aliases, WAVs, synthesis units or recording scripts are changed by previewing. Downloading is user-initiated; the backend does not write arbitrary user-supplied output paths.

## Safeguards and limits

- The new preview endpoint has type checks for the existing voicebank path, integer budget 0–10000, and optional existing-reclist path. JSON request size is bounded to 8 KiB; a foreign Origin is rejected.
- It clears the cached report **before** handling each new preview. A failed preview cannot download a previous analysis result.
- The browser hides stale results as soon as the path or budget changes and renders all voicebank-derived strings with DOM `textContent` rather than HTML injection.
- The localhost GUI has a **single process-wide in-memory report**, not separate per-browser-session report caches. Do not expose this loopback service to untrusted local web pages or concurrently use multiple tabs as though each has separate projects. CSRF and same-origin protections should be strengthened in later production packaging.
- This v19 feature inherits the current v18 **Mandarin OTO** coverage limitations and does not support arbitrary user-native script or independently validate phoneme boundaries. Distinct PCM is not distinct singing performances. An empty list does not prove a complete voicebank.
- No new acoustic model, threshold, recording strategy or synthesis-unit split is introduced.

## Use

Run `phonoweave-gui`, select an existing voicebank, open the **Recording Plan / 补录方案** link from the analysis home page, preview the suggestions, then click the text or JSON attachment. The original `phonoweave plan-recording` command remains available and unchanged.

## Validation

`tests/test_recording_plan_gui_v19.py` covers page routing, refused premature download, preview and content-disposition downloads, stale-cache invalidation on errors, input rejection, foreign-origin rejection and DOM text-safety. Full repository CI is required to determine compatibility with the existing GUI routes and analysis suite. This is software integration testing, not a demonstrated improvement in recorded singing quality or elapsed recording time.
