# Experiment 1: recording budget versus held-out acoustic coverage

## Hypothesis

Speaker-adaptive sample allocation could improve retained **acoustic representativeness** at a fixed recording budget compared with a deterministic phonetic-context balancing policy and seeded stratified-random selection. The hypothesis may fail; no advantage is assumed. This exploratory metric is not synthesized-audio quality.

## First supported study

Existing Mandarin OTO-segmented fricative WAVs with recognized `s`, `sh`, `x` plain/rounded contexts. The budget unit is one physical recorded WAV, not an alias or segment. Byte-identical WAVs across paths share a SHA-256 group. For each recognized segment, a fixed normalized vector of spectral moments, slope, high-band ratio and duration is derived from the core 20–60% consonant region. The evaluation is conditional on the availability and reliability of OTO annotations.

For each of five fixed random seeds, a 25% holdout is created at **WAV group** level using context metadata, without inspecting holdout acoustic values. Remaining WAVs form the acquisition pool. Every strategy has the same budget and the same candidate phonetic metadata:

- `fixed_balanced`: deterministic balancing by onset/context cell.
- `stratified_random`: seeded random acquisition weighted towards underrepresented context cells.
- `adaptive_observed`: uses balancing plus the observed spread among **previously selected** WAVs, and unrepresented finals; acoustic features are revealed to this policy only after acquisition.

Budgets are 25/50/75/100% of acquisition-pool WAVs. At each budget the held-out pool is fixed across strategies. Report both held-out context-cell coverage and mean nearest-neighbor normalized acoustic distance within the same onset/context cell; penalize entirely missing cells separately. Aggregate across folds and retain all per-fold scores. Scaling constants and missing-cell penalty are fixed before looking at held-out data, not fitted on evaluation audio.

## Avoiding leakage

A source WAV may contain multiple OTO aliases or phonetic segments; all segments from it stay in the same split and count against one recording budget item. The selection policy receives only each candidate's context/final metadata; it gets acoustic features of a WAV only after choosing that WAV. The test set is not in the acquisition pool. Fold means are descriptive and correlated across folds; they are not cross-speaker confidence intervals. A deterministic source-profile-derived decision used before splitting would leak evaluation evidence, so this experiment does **not** reuse full-corpus SpeakerProfile decisions in the acquisition policy.

## Limitations

A retrospective experiment with full preexisting recordings is not prospective recording-list generation: the candidate metadata is known but cannot guarantee unrecorded WAV audio. Exact SHA-256 grouping does not detect transformed duplicates. OTO quality and singer/pitch differences can affect feature vectors. The fixed context-cell and Euclidean acoustic-distance metric is a proxy, not proof of substitution quality, acoustic phonemic equivalence or actual singing-synthesis coverage. Multi-speaker and synthesis-based assessments remain required.

## Usage

```bash
python -m pytest -q
python -m phonoweave.recording_budget_benchmark_cli /path/to/voicebank --output /path/to/report.json
```

No voicebank files or synthesis inventories are modified. Run first on a voicebank with sufficient independently recorded `s/sh/x` WAVs and OTO timing metadata. The CLI rejects insufficient or non-stratifiable data instead of fabricating success.
