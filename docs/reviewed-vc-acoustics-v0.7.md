# v0.7 — Reviewed VC acoustics: bounded diagnostic and next experiment

Based on the **six user-reviewed LynneQingD4 excerpt boundaries**, not on the old OTO midpoint.

## Reproduce

```bash
phonoweave reviewed-vc-probe \
  --pack /path/to/LynneQingD4_VC_Interactive_Review_v07 \
  --annotations review_annotations_completed_v07.csv \
  --output /outside/new_vc_diagnostics
```

The command verifies source WAV SHA-256, source identity and exact six-slot manual review. It samples five 16-ms windows *within each manually marked interval* and reports normalized 12-band log-power spectrum summaries. For each matching right-hand surface label, it computes exploratory pair distances of the i / ir / i0 classes, including ±6-ms start/end shifts on both clips (81 variants). It refuses to overwrite original source folders or existing outputs.

## Descriptive observations (user-provided source; no WAVs committed)

Author marks in the 600ms excerpts:
- i→sh: 219.8–263.8 ms; ir→sh: 234.8–282.7 ms; i0→sh: 226.1–289.5 ms
- i→m: 236.6–290.2 ms; ir→m: 207.0–259.7 ms; i0→m: 211.8–254.5 ms

All six rows have identity and excerpt hash matches. The /sh/ timecourses have rising high-frequency energy across these intervals while /m/ does not show the same fricative cue. Several /sh/ spectral distances vary noticeably under ±6ms shift; do not interpret their ranking as split/share efficacy.

**No-go:** One source excerpt for each label/right-hand target, no independent repeat sessions or second adjudicator, and no matched recording-cost rendered synthesis. Neither acoustic distance nor the user's single human annotation is sufficient to approve automatically split/merge. Speaker-specific model, inferential effect size and population claims remain blocked.

## Next minimal acquisition protocol

Record each of `li → ma`, `shi → ma`, `zi → ma`, `li → sha`, `shi → sha`, `zi → sha` three times in each of two independent-day sessions (S01 training, S02 held-out), total **36 new takes**. Freeze comfortable pitch, mic setup and score/timing before first capture. Independent takes cannot be derived from duplicated original WAV snippets. Use S01 for model selection, and test S02 *only* after freezing. Measure actual recording time as budget, verify boundary annotations and compare phoneme/context-aware baseline with speaker adaptive strategy; test real rendering and blinded listening before efficacy claims.

The local data and recording plan are provided to the author separately and not committed to the repository.

Tests: `pytest -q tests/test_reviewed_vc_probe.py`.
