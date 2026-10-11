# Calibration v1.2 experimental onset detection

Two user-provided WAV files from 2026-10-09, at 48 kHz, 5.205 s each,
show sustained high-frequency frication **preceding** the manually labeled
anchor in both target repetitions of xi and shu. Human annotations likely
mark fricative-to-vowel transitions, not the beginnings of frication; do
not compute onset MAE against those questionable gold marks.

The browser captures mono samples using AudioContext but emits the
beep from an independent oscillator to the speakers. The recording has
no guaranteed cue channel or persisted precise audio-clock timestamps.
The old beep reconstruction may mistake speech harmonics for cues.
v1.2 uses 250 ms + (token_index+1)*833.333 ms as a timing prior only.

- 20ms Hann FFT, 4ms hop, 2.5–10 kHz high-band energy
- high-band rise >= 12dB against initial-window baseline
- high-band fraction >= -12 dB relative to full-band RMS
- absolute high-band >= -90 dBFS
- minimum 40 ms consecutive presence, short clicks rejected
- search -350/+500 ms relative to nominal beat
- abstain if frication is already present at search edge
- report manual-anchor disagreement, not misleading MAE

Run:
```bash
python -m pytest -q
python -m phonoweave.calibration_transition_eval_cli \
  ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Output: analysis/calibration_transition_onset_v1.2.json.
No existing recordings or manual labels are modified.

Single-speaker, same-session exploratory study only; no held-out
validation. JS timers may drift; future recordings need a reference
track or audio-clock metadata.
