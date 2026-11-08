# VC boundary feasibility gate (v0.5)

This milestone **stops the automatic split/share evidence pipeline at the first point where OTO timing may be misleading**. It is a manual-audit stimulus generator, not a new spectral split classifier or another inventory count.

## Run

```bash
phonoweave vc-boundary-gate /path/LynneQingD4.zip \
  --labels examples/lynneqing_d4_confirmed_labels.json \
  --right m --right sh \
  --output /outside/boundary_review_01 \
  --plots
```

The CLI creates a **new** directory, keeping all original WAV/OTO files unchanged. It verifies the entire source OTO SHA-256 against the author-confirmed alias mapping, and includes:
- 600ms native-rate, unchanged-level excerpt WAVs for `i`, `ir`, `i0` before/after each OTO-derived anchor;
- relative-to-anchor 16ms RMS and spectral-centroid trajectories every 20ms across ±160ms;
- optional PNG trend charts (requires separately installed `matplotlib`);
- source hashes, OTO line numbers, unchanged alias strings, and an empty manual boundary annotation CSV.

The reference point at 240ms within each excerpt is **the midpoint between OTO overlap and preutterance**, not a verified phonetic boundary. It is used only to make the listening review reproducible.

## Actual feasibility finding on supplied LynneQingD4

The original ZIP contains 25 WAVs, 1,935 OTO records and 1,124 VC-shaped aliases. The earlier v0.3 extraction successfully measured 830 16ms snapshots; **that was a measurement count, not 830 validated transitions**.

For `D4i sh`, `D4ir sh`, `D4i0 sh` (the last value is `i0`, not a zero vowel), the actual measured spectral centroids **at the candidate anchor** were 340, 423 and 324 Hz, respectively, and **60ms later** they were 4,493, 3,606 and 3,879 Hz. The first descriptive 2kHz crossings appear at +20, +40, +40ms. This is consistent with high-frequency frication energy becoming salient only after the old reference point. It is *not* proof of a precise frication onset: listening and boundary annotation remain required.

All three `m` classes can be inspected in `vcsection2.wav`. The three `sh` classes are **not** from the same file: `ir sh` is sourced from `ingi0ir.wav`, while the other two are on `vcsection3.wav`. Neither group contains independently repeated recording sessions.

## Research go/no-go

**NO-GO for automatic split/share claims on the current OTO-window proxy**. Do not reuse those single-frame features as speaker-adaptive unit labels. The very next task is to annotate true CV/VC boundary intervals of these six listening clips, then acquire independent recording takes of exactly matched targets. Re-evaluate with held-out takes and real rendering before publishing any efficacy result.

No user-provided audio, generated excerpt, personal source paths, render outputs or recordings are committed here. Only code, tests and documentation are versioned.

Run `pytest -q tests/test_vc_boundary_gate.py`; on the supplied ZIP this command generates a separate, user-private audio review package.