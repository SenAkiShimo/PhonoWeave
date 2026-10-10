# Manual Anchor v0.7 — spectrogram and free timeline

This experimental release responds to manual-labeling issues encountered on the 2026-10-09 46-prompt Mandarin calibration.

## New

- Shared waveform + FFT magnitude spectrogram (0 to the lesser of 8 kHz or Nyquist).
- Timeline position slider, zoom slider, and Reset; clicking either plot marks the current time.
- Per-recording cached FFT data; moving the timeline reuses the computed spectrum.
- Backend accepts anchor positions anywhere in the physical recording and adds `outside_expected_slot` for review. It still rejects NaN or positions beyond the actual recording duration.
- Neutral labeling guidance; no conversational development notes in the interface.

## Run

```bash
cd ~/Downloads/PhonoWeave
git fetch origin
git switch --track origin/feature/manual-anchor-spectrogram-pan-v0.7
source .venv/bin/activate
python -m pytest -q
python -m phonoweave.calibration_manual_labeler \
  ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Marking continues in the original JSON file:
`analysis/calibration_manual_anchor_labels_v0.2.json`.
The 46 WAV files remain unchanged.

## Limitations

A dark or difficult sonorant boundary may still require listening; a spectrogram is evidence, not ground truth. The 512-sample FFT prioritizes quick interactive inspection over detailed acoustic measurements. Very early/late out-of-slot marks are **permitted but flagged**, and should not automatically enter trusted evaluation metrics.

Changes have not yet been validated on a user's machine; manually review interaction in Chrome/Safari and run tests before merging.
