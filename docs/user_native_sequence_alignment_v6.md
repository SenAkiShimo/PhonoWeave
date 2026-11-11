# PhonoWeave v6 — User-native sequence alignment (research prototype)

**Status:** Synthetic tests only. No independent real-phone boundary validation and no new recording scheme is authorized.

The input is the frozen v5 **candidate_segment_unit_hints.csv**, plus a separately authored **complete ordered list of the user's own units for each original WAV**. We do not inspect `oto.ini`, OTO windows, or reference boundaries to select candidate region boundaries, determine pronunciation order, or fit costs. We never require IPA/G2P/dictionary conversion. Symbols like `i`, `i0`, `i2`, `ir`, `a1`, and Chinese characters are opaque strings.

## Why v6 is different from v5

v5 classified acoustic events individually; not every change-point is an independent phone. v6 aligns an **ordered sequence** to **contiguous regions**, permitting:

1. Skip an acoustically irrelevant change region.
2. Map up to four neighboring v5 regions to one user unit (many acoustic regions to one unit).
3. Associate a region with two adjacent user units without inventing an interior boundary; this is explicitly `unresolved_shared_interval_NOT_a_phone_boundary`.
4. Abstain when evidence or the alignment path is ambiguous.

All algorithm costs (`SKIP_COST`, `SHARED_UNIT_PENALTY`, etc.) are fixed exploratory heuristics and have **not** been tuned or validated on independent phone boundaries. Top-two path difference is **not a probability**. The sequence itself **must not** be automatically populated from OTO aliases or held-out boundaries: it comes from an independent recording script/transcription supplied by the author.

## Inputs, provenance and reproducibility

Create `sequences.json` in the format of `examples/user_native_sequences_v6.example.json`. Each entry contains `bank`, `source_wav`, a frozen `source_pcm_sha256`, `provenance: independent_recording_script`, and a complete array `units` in pronunciation order. Do not include IPA unless those happen to be the user's actual strings; the method does not translate symbols.

The v5 input is either the `results/` directory containing `candidate_segment_unit_hints.csv`, or a ZIP archive with `results/candidate_segment_unit_hints.csv`. Inference validates PCM SHA-256 consistency between input records and user-supplied sequences. It refuses existing output directories, duplicate source keys, overlapping candidate intervals, missing source rows, hash mismatches, and non-independent sequence provenance. Original WAV archives and OTO data are not modified.

```bash
python -m phonoweave.native_sequence_alignment align \
    --v5 /path/to/v5-results-or-zip \
    --sequences /path/to/sequences.json \
    --output /path/to/NEW_v6_output
```

`alignments.json` records the ordered alignment, span start and end, skipped-region count, path cost, nonprobabilistic alternative-path margin, source hash, and each unit's unverified/abstaining status; `summary.json` reports candidate vs. unresolved counts.

## Independent validation phase

Only after freezing and generating `alignments.json`, create **independent, human-audited** `gold.json` with the format of `examples/user_native_gold_v6.example.json`. Its unit indices, symbols and PCM hashes must match the frozen sequence exactly; timestamps must be manual annotation **not used at any point in inference**. A gold boundary must represent the intended unit onset/offset (and definition should be fixed before annotation); OTO anchor timestamps are not a substitute.

```bash
python -m phonoweave.native_sequence_alignment evaluate \
    --predictions /path/to/NEW_v6_output \
    --gold /path/to/gold.json \
    --output /path/to/NEW_v6_eval
```

Report endpoint MAE on nonabstained units **together with** abstention/coverage. Use at least one separately held-out speaker/bank or recording template that did not influence event detector thresholds, alignment heuristics or source training; freeze protocol before collecting boundary annotations. The existing v5 OTO comparisons are diagnostics, not independent ground-truth phone segmentation. Do not call this automatic discovery of `i/i0/i2`, automatic phone recognition, or validated boundaries.

## Future work

- Compare against order-only duration baselines, acoustic-only change-point baselines, and shuffled-script negative controls.
- Use independent manual corrections to quantify missed boundaries and false positives, then expand to multi-speaker recordings.
- Only after synthesis A/B evidence decide whether candidate distinctions deserve recording-scheme split/merge.
