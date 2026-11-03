# Controlled /sh/ pilot: frozen S01 → S02 acoustic proxy evaluation v1.3

The v1.2 capture grid freezes 4 exact plain/rounded final pairs
(`a/ua`, `ai/uai`, `an/uan`, `ang/uang`), 3 takes per final
per session, and 2 distinct-date sessions (48 WAV targets total).

This v1.3 evaluation adapter uses **only** an audit-ready manifest:
all 48 WAVs must satisfy format, SHA-256, session provenance and manual
onset/CV annotation checks. Incomplete captures emit `insufficient` with
no budget or outcome curve. It makes no silent timing imputations.

## Commands

```bash
# Run on the controlled-corpus research branch (not main):
python -m phonoweave.synthesis_relevance_controlled_corpus plan \
  --output /path/to/pilot/manifest.json

# Record all S01 and S02 targets on independently dated sessions;
# annotate friction onset and consonant–vowel transition manually.
python -m phonoweave.synthesis_relevance_controlled_corpus audit \
  /path/to/pilot/manifest.json --root /path/to/pilot

# Evaluate without changing audio or the existing manifest:
python -m phonoweave.synthesis_relevance_controlled_evaluation_cli \
  /path/to/pilot/manifest.json --root /path/to/pilot \
  --output /path/to/new_spectral_proxy.json
```

Evaluation refuses existing outputs. The JSON includes source manifest
and per-take audio hashes, frozen train/test roles, train-only utility,
held-out per-take spectral distances and full 0–4 split-identity budget
curves for alphabetic, prespecified linguistic, and adaptive policies.

## Extraction and leakage discipline

- Fixed 20 ms Hann-window log-normalized spectral signature sampled at the
  **middle of the manually labeled frication region**, requiring safety
  margins from the annotated CV boundary. No preutterance-based or
  assumed vowel boundary is substituted.
- **S01 only** supplies donor centroids and the ranking of context
  pair splits. S02 WAVs are never read during policy selection.
- **All 24 S02 WAVs** are evaluated under all 3 strategies for every
  0–4 split-pair budget. Fixed/linguistic ranks are declared before data.
- Unit of cost is **one additional pair-specific donor identity**, not
  one extra line, distinct session or measured capture duration.
- The spectral-centroid error is a simplified reconstruction proxy. It is
  **not** DDSP/UTAU/OpenUtau audio rendering or listening quality. This
  milestone does not assert a positive effect or justify production aliases.

## Additional validation before a scientific claim

Independently annotate & review each recording, check clipping and pitch
drift, quantify device/room variance, freeze calibration before collecting
new sessions, recruit other speakers, add real duration costs and use
rendered singing A/B with blinded listeners.

Synthetic unit tests show software invariants, **not** any acoustically
observed finding.
