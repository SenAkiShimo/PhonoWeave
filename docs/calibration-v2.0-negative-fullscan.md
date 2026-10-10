# Negative-control full-scan audit

The v1.9 screening yielded 9 independent category-labeled negatives. One breath clip triggered frication detection. The v1.9 detector was evaluated at a single nominal 1250 ms cue, limiting the measured interval.

This audit scans overlapping windows spaced by 200 ms, retaining the original v1.4/v1.5 acoustic detector thresholds. Overlapping-window estimates are clustered within 80 ms to report approximate distinct events. The first and last portions of each recording may not have complete search-window support and must not be interpreted as fully inspected.

Run:
```bash
python -m pytest -q
python -m phonoweave.calibration_negative_fullscan_cli
```

Output: `negative_controls/negative_fullscan_v2.0.json`.

The source clips are unchanged. Category labels are self-reported and require manual acoustic adjudication. Candidate counts are preliminary trigger indicators, not validated false-positive rates. Review the breath clip before changing acoustic rejection thresholds.
