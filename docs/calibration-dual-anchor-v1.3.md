# Calibration dual-anchor annotation v1.3

This branch adds two **independent** marks for every existing manual-annotation target:

- **Consonant onset** (green): the defined onset acoustic event (sustained frication begins for sh/x/f/h, release/frication event for affricates/stops, sonorant onset for m/n/r).
- **CV boundary** (orange): the transition from consonant to vowel, when vowel-like voiced/formant structure begins to dominate.

The original 32-label JSON (v0.2) is **not silently reclassified** as either event, because some old placements actually mark the CV boundary. It stays read-only, visible as a gray reference line.

New annotation path (created only on first save):

`analysis/calibration_dual_anchor_labels_v0.3.json`

The v0.2 path remains unchanged:

`analysis/calibration_manual_anchor_labels_v0.2.json`

The page offers a mode selector. Click either waveform/spectrogram to set the **selected** anchor only; other anchor remains displayed and untouched. The uncertain and clear controls act only on the selected anchor. Progress counts an occurrence as complete when **both** anchor types are filled; their statuses remain independent. For a previously labeled occurrence, the explicitly named **Copy legacy to selected anchor** button copies the legacy timestamp/status/provenance into the currently selected field only if that field is empty, so it must be reviewed as the chosen event.

Do **not** take old v0.8/v1.2 error figures as onset MAE until onset marks are independently reviewed in the dual schema. Future evaluation should read the v0.3 JSON and calculate separate onset / CV metrics rather than mixing anchor definitions.

Run:

```bash
cd ~/Downloads/PhonoWeave
git fetch origin
git switch --track origin/feature/calibration-dual-anchor-v1.3
source .venv/bin/activate
python -m pytest -q
python -m phonoweave.calibration_manual_labeler \
  ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813
```

Test in Chrome/Safari for two different-colored markers on waveform and spectrogram and inspect the new file after saving. These browser behaviors are pending real-machine verification. Do not merge without confirming.
