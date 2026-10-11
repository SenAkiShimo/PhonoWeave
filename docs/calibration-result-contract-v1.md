# Calibration result contract v1 (experimental)

The exporter emits `analysis/calibration_results_v1.json` for a recorded calibration session. It is a **research-only** boundary between elicited acoustic calibration and future downstream applications. It does not modify `SpeakerProfile`, `SynthesisInventory`, OTO timing, or the current recording-list generator.

Only Mandarin x/sh elicited CV targets are evaluated. Unsupported recorded consonants are explicitly listed with reasons. Missing recordings are counted, not treated as absent acoustic events. Every modeled occurrence carries a nominal cue, detector method/status/reason, raw onset and CV timestamps, and independently named experimental backtrace and selected-onset fields. The fields are all milliseconds from WAV start. Unpaired events have no selected onset. Every candidate requires manual review.

Provenance fields include relative WAV path, sample rate, SHA-256 digest, protocol name/version and model identifier. Human reference annotations are intentionally never read during prediction export. The contract is not a calibration accuracy report. Nominal cue timing is not sample-accurate, and the model is not validated on non-x/sh consonant classes.

Run:

```bash
python -m phonoweave.calibration_results_cli ~/Downloads/PhonoWeaveCalibration/20261010_175208_925611
```

Consumers must check `schema`, `schema_version`, `model.maturity`, class support, and per-occurrence status. No downstream synthesis-inventory decision should use a `research_only` candidate as a verified singer-specific acoustic measurement without an explicit review and validation policy.
