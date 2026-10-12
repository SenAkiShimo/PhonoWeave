# Synthesis Relevance v0.8: offline unit-merging cost proxy

## Purpose and limitations

Test whether borrowing a plain or rounded Mandarin `sh` recording unit from the **wrong context** creates greater objective mismatch than borrowing a held-out unit from the **correct context**. This is a stricter question than acoustic classification, but it is **not yet a real OpenUtau synthesis experiment**. It evaluates donors against original held-out WAV signal signatures and a frozen target-right spectral boundary; no song note, resampler or rendered output is invoked.

## Design

- Use already-recorded OTO/WAV data without modifying or exporting a sound file.
- Candidate OTO interval is `[offset, offset+preutterance]`. Minimum 55 ms; sample the provisional donor **core** at 42% of preutterance (at least 14 ms from OTO start) and the frozen target **late** location 16 ms before preutterance. Both signatures use 20 ms Hann windows, 48 log-normalized spectral-energy bands between 1.5 kHz and min(11 kHz, 46% Nyquist-equivalent sample rate). A separate RMS dB mismatch is recorded.
- Hold out each target. Select candidate donors from the **same subbank** and different physical WAV, separately for same-context and cross-context borrowing. Match duration before spectral scoring, choose equally many donors per arm (up to three), then use the median result per arm.
- Compute (i) donor core against real target core log-spectrum as a **reconstruction proxy**, (ii) donor core against the fixed real target late spectrum as a **junction proxy**, and (iii) donor core-vs-target-late RMS dB mismatch.
- Subtract same-context scores from cross-context scores: positive delta indicates larger mismatch for cross-context reuse.
- Aggregate contrasts to one observation per target physical WAV before reporting mean delta, percentile bootstrap confidence interval and two-sided WAV-level sign-flip p (1999 draws). Never treat multiple OTO aliases on one physical WAV as independent trials. Report per-subbank statistics and donor-duration mismatch; these are important covariate checks.

## Interpretation

A positive core reconstruction contrast that persists across subbanks is consistent with a **measurable penalty of context merging under this offline metric**. A positive junction contrast provides supplementary support, but it could reflect an intrinsic context-dependent spectral relation between the target's fricative and following vowel, not necessarily audible seam damage.

The two controls are not acoustically exchangeable automatically: donors differ in following vowel, word list, pronunciation context, sample timing and OTO recording setup. The target core reference is extracted by the same imperfect OTO timing convention and may include transition into a vowel. No claim of necessity, perceptual detectability, rendered singing degradation, generalization across speakers, or actual optimal recording inventory follows directly.

## Run

```bash
cd ~/Downloads/PhonoWeave
source .venv/bin/activate
python -m pytest -q
python -m phonoweave.synthesis_relevance_unit_merging_cost_cli \
  ~/Downloads/phonoweave_voicebanks.json \
  --base sh \
  --output ~/Downloads/phonoweave_sh_unit_merging_cost_v0.8.json
```

Inspect `metrics.delta_reconstruction`, `metrics.delta_seam` and `duration_match_summary`. All WAV inputs remain unchanged; only JSON output is created.
