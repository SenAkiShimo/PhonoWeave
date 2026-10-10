# Independent negative-control screening v1.9

Adds a separate local recording workflow at `/negative-controls` in the PhonoWeave GUI. The capture protocol includes standalone /a/, /i/, breath, silence, room noise, and handling noise as distinct categories, with environment and input-device type. Browser echo cancellation, noise suppression, and automatic gain control are requested off; actual boolean settings are recorded when reported by the browser. Device identifiers and names are not persisted.

Audio clips are 3-second mono PCM WAV takes, stored under `~/Downloads/PhonoWeaveCalibration/negative_controls`, outside existing positive calibration session directories. No target syllables or audible prompts are inserted into the WAV.

```bash
phonoweave-gui
# open /negative-controls
python -m phonoweave.calibration_negative_controls_cli
```

The analyzer runs the unchanged v1.4 strict and v1.5 shoulder detectors in a standardized 1.25-second local evaluation window. Negative labels are self-reported, so trigger fractions are screening indicators rather than independently verified false-positive rates. The pipeline does not tune thresholds automatically, change original recordings, or infer cross-device validity.

**Limitations:** actual input device category is selected manually; local mobile browser recording paths may differ from standalone phone voice recorder apps. Timing of JS setTimeout controls take duration, not precise acoustic onset. Neither the legacy beep alignment nor a precise audio-clock reference is solved in this branch; recorded event and device metadata is partial. Distinct validated positive datasets and manually reviewed negatives are required before production specificity claims.
