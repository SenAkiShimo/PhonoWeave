# OTO segment yield and WAV-duration cost audit (v0.6)

## Motivation

CVVC recorded files contain multiple OTO-timed aliases and potentially several independently extractable consonant segments. Prior v0.1–v0.5 recording-budget comparisons use the physical unique WAV as an indivisible acquisition unit. This correctly prevents splitting one file between train/test folds but does **not** equalize usable phonetic evidence or actual recording effort across strategies and voicebanks.

## Frozen experiment audit

For each existing eligible WAV, the audit derives:
- Number of unique OTO-extracted target `s/sh/x` acoustic observations (not the total OTO alias inventory).
- Full WAV duration from PCM headers, including all unmodeled consonants, vowels, silence and padding.
- Segment yield and duration burden for every originally selected WAV at 25%, 50%, 75% and 100% WAV budgets.

The audit replays the exact v0.2 five-fold disjoint partitions and three frozen selection policies, without changing any scoring feature, distance metric or selection coefficient. All segments within one chosen WAV are acquired together. Equal WAV counts can correspond to unequal eligible-phoneme evidence and unequal audio duration. The audit reports each as a separate descriptive quantity, not as a corrected accuracy score.

## Scientific interpretation

OTE alias counts can include multiple names referring to one acoustic realization and must not be treated as independent acoustic tokens. The source acoustic extractor groups coincident segments under shared OTO timings and discards ambiguous labels. The eligible segment count therefore reflects the current extraction method and its segmentation assumptions, not total realizations in the original recording.

Physical WAV duration is only a surrogate of actual recording burden. It includes non-target phones and rests, while repeated attempts, setup time, pronunciation difficulty and take selection are not measured.

The audit intentionally does **not** change the WAV-count evaluation or claim equal-duration performance. A following preregistered experiment can compare policies under fixed actual-time budgets and macro context/segment coverage with new independent voicebanks.

## Run

```bash
python -m pytest -q
python -m phonoweave.recording_budget_cost_audit_cli \
  ~/Downloads/phonoweave_voicebanks.json \
  --output ~/Downloads/phonoweave_recording_cost_audit_v0.6.json
```
