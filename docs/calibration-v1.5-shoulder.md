# Calibration v1.5: weak frication shoulder audit

Observed v1.4 (four labeled x/sh fricatives): onset MAE 11.746ms,
CV boundary MAE 7.759ms, all CV references legacy-imported.
The first shu onset differs by +34.589ms from manual mark, while the
other three onset absolute errors are below 9ms.

An exploratory inspection of raw `016_sh_rounded.wav` finds a gradual
rise in high-band energy around 1,170–1,180 ms, while the v1.4 strict
high-band fraction gate admits a 40ms confirmed core starting at 1,216 ms.

v1.5 keeps the exact v1.4 strict core; for a confirmed core only, it
traces back at most 80ms through contiguous high-band frames above
both -90dBFS and local baseline +10dB. This avoids removing the core
noise-ratio gate. The result is an *experimental shoulder candidate*
alongside the unchanged v1.4 result, **not a silent replacement**.

Run:
```bash
python -m pytest -q
python -m phonoweave.calibration_frication_shoulder_cli \
  ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

The original WAV files and manual JSON anchors are not modified.
The report includes per-occurrence timing and paired MAE metrics. The
same-session errors cannot establish generalization; additional
held-out recording sessions are needed.
