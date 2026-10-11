# Experiment 1: independent voicebank replication (v0.5)

This exploratory replication uses unchanged v0.2 five-fold disjoint WAV partitions and v0.3 100-repeat random-null comparisons separately on multiple Mandarin voicebanks, ideally from at least three independently identified singers. The first LynneQing voicebank is a pilot; collect additional voices before drawing cross-speaker conclusions.

## Manifest

```json
{
  "schema": "phonoweave.voicebank_replication_manifest.v1",
  "voicebanks": [
    {"bank_id": "pilot01", "speaker_group_id": "singer01", "path": "/absolute/path/to/first/voicebank"},
    {"bank_id": "validation02", "speaker_group_id": "singer02", "path": "/absolute/path/to/second/voicebank"},
    {"bank_id": "validation03", "speaker_group_id": "singer03", "path": "/absolute/path/to/third/voicebank"}
  ]
}
```

One voicebank per speaker group. Do not assign separate IDs to multiple pitches, appends or phonation styles recorded by the same singer. Paths are read locally and omitted from output; WAV hashes and full recordings are not published.

## Protocol and exclusions

For each bank, extract OTO-labeled Mandarin `s/sh/x` segments from existing WAVs and deduplicate on WAV SHA-256. Banks need at least 20 eligible WAVs. Fewer valid WAVs, missing folders, schema failures or duplicate audio across previously evaluated banks produce an explicit exclusion record. The report records counts of skipped and ambiguous segments, warnings, covered folds, and any train/test metadata-cell gaps. An identity group is **declared**, not biometrically verified.

Never pool individual WAV examples across speakers for estimating the result. Evaluate the unchanged v0.2 and v0.3 pipelines separately for each bank; fail if the first random draw or adaptive score no longer matches v0.2. Report policy deltas at 25/50/75/100% recording budgets, the random-baseline sensitivity distribution, and unweighted macro summaries with each speaker counted once.

The first bank alone is not enough for a cross-speaker result. Two successful independent speakers are an initial replication, not a robust multi-speaker conclusion. Comparability remains exploratory: differing OTO quality, language coverage, acoustic balance, and recording sessions affect results. In particular, WAV-count budget is not equalized recording effort, the metric is still a spectral proxy (not synthesis quality), and no acoustically re-encoded duplication is detected by SHA-256. No adaptive coefficients, feature normalization or scores are tuned.

## Run

```bash
python -m pytest -q
python -m phonoweave.recording_budget_cross_voicebank_cli \
  ~/Downloads/phonoweave_voicebanks.json \
  --repeats 100 \
  --output ~/Downloads/phonoweave_cross_voicebank_v0.5.json
```
