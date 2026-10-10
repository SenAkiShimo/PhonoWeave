# Calibration v1.8 carrier trigger inspection

Two carrier-window events need manual review. Prompt 7: cai a cai a, with a candidate at 1612 ms (304.667 ms before the nominal carrier cue); the preceding cai may overlap the -350 ms search interval. Prompt 27: a bian a bian a, with a candidate at 992 ms (91.333 ms early), occurring near the first carrier vowel and therefore not explained by a preceding consonant token.

Run python -m pytest -q, then:

    python -m phonoweave.calibration_carrier_trigger_review_cli ~/Downloads/PhonoWeaveCalibration/20261009_225541_655813

The exported clips are stored in analysis/carrier_trigger_review and preserve offsets in the companion JSON report. Review spectrogram and audio before labeling any event as a false positive. Existing WAVs remain untouched.
