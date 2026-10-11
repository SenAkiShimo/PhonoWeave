# Calibration evidence adapter v1

The adapter converts the **research-only** `phonoweave.calibration_results` v1.0.0 manifest and separately authored `calibration_dual_anchor_labels_v0.3.json` into a read-only, auditable `phonoweave.calibration_evidence` v1.0.0 report. It does not change any speaker-profile, synthesis-inventory, OTO timing, WAV audio, or original manual annotations.

Run:

```bash
python -m phonoweave.calibration_evidence_adapter_cli ~/Downloads/PhonoWeaveCalibration/20261010_175208_925611
```

The adapter verifies the session identity, recording-relative paths, sample rates, WAV SHA-256 hashes, expected target occurrences, model maturity, timestamp order, annotation source and anchor status. A usable *manual timing pair* requires separate in-slot `consonant_onset` and `cv_boundary` marks with `status=ok` and `source=manual_v0.3`. Missing, uncertain, imported or out-of-slot labels remain review-required. Timing contradictions are reported for review and cannot become a valid manual pair. Manual marks are **not** treated as independent inter-annotator adjudication.

Every output observation records the acoustic prediction, both manual marks if accepted, signed timing discrepancies when meaningful, and explicit `speaker_profile_admissibility=blocked` and `inventory_decision_admissibility=blocked`. A complete annotation pair does **not** establish phonemic realization separability, synthesis relevance, speaker identity, or generalizability. The adapter intentionally provides no automatic profile-merge/split functionality.

Use a session's existing `analysis/calibration_results_v1.json`, generated separately with `calibration_results_cli`. Only elicited Mandarin x/sh observations are covered. Current outputs are research artifacts and require independent device/speaker validation before downstream policy design.
