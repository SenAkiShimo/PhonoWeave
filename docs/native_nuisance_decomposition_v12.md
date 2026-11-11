# PhonoWeave v12 — paired-bank acoustic variation decomposition

**Status: exploratory software and retrospective acoustic evidence; not phoneme identification or a deployable style-independent representation.** No automatic phone alias, reclist, oto.ini, voicebank, or synthesis inventory modifications. The output cannot establish whether an acoustic difference should be collapsed or preserved for synthesis.

## Motivation

v10 found stable-looking global modes that largely captured the order of spectral intervals. v11 found that the apparent three-way subdivision at the last interval was not replicated when the two recording styles were trained independently (validation ARI 0.11065). Blindly normalizing all carrier/context differences would risk deleting useful coarticulation. v12 instead tests one **narrow, reversible nuisance hypothesis**: within the same historical recording template, can a bank-specific *translation* reduce systematic style/pitch-bank mismatch **without changing within-bank pairwise acoustic distances**?

It does **not** assume any segment is a phone. Source clip names such as `a_to_sh` are metadata for *pair matching and family holdout only*, not independently validated phonetic labels. The 7 bank names represent 7 pitch/style subbanks, not 7 independent people.

## Input and procedure

- Read frozen v9 `motif_regions.json`, `raw_spectral_embeddings.npz` (48D log-mel representations) and `summary.json`, optionally verifying v10's region times/PCM hashes and source archive SHA. Or read an independent ZIP of mono PCM16 full WAVs through v10's audio-only extractor; no IPA, dictionary, OTO timestamps or user-unit sequence are needed.
- Reject non-finite features, malformed source PCM hashes, repeated acoustic-region IDs and the same PCM digest assigned to distinct clip families. Refuse overwriting output.
- Stratify by *automatically detected region ordinal*, not claimed phone position. Keep every carrier family across banks within one fold (v10's deterministic 60/20/20 train/validation/audit group split). Normalize using **train-only** shrinkage-scaled statistics.
- For each training carrier `g`, compute the mean normalized vector per bank `b`. Calculate the carrier-specific center averaging banks equally, then a bank effect `d[g,b] = mean(g,b) - mean_over_banks(g,b)`. Average these differences over eligible paired training carriers to obtain one 48D offset `offset[b]` per bank. Require ≥6 independent training carrier families per bank, ≥3 banks, and ≥2 styles.
- The candidate `x_corrected = x - λ × offset[bank]` uses one of five fixed strengths `λ∈{0,.25,.5,.75,1}`. Candidate ranking is based on **validation carriers only**: same-carrier top-1 retrieval across *different-style bank pairs* plus a matched-to-distractor spectral distance ratio. This supervised *metadata pairing* is a nuisance diagnostic, **not** self-supervised recognition of phonemes. Do not call the task language-independent phone discovery.
- A proposal passes a **development-only** screen if the validation top-1 gain is at least 0.025, relative distance-ratio gain is at least 0.04, and an exploratory matched-identity null p-value is at most 0.05. The null randomly permutes bank identities **within each training carrier**, refits the bank offsets for each of 32 seeded trials and permits each null trial its own best strength on validation. All thresholds are **retrospective research choices, not calibrated statistical guarantees**.
- Exact preservation of within-bank squared Euclidean distances follows algebraically because a given bank receives one shared translation. Test it numerically and refuse a proposal if the maximum distance change exceeds 1e-7. This does not preserve every cross-bank/context or perceptually important distinction.
- Only after the validation strength and null comparison are fixed, compute **descriptive** metrics on the unused carrier-family audit fold. No audit outcome is used to select λ, thresholds or bank offsets; descriptive audit failures must not be used to retrospectively retune the algorithm. The v11 stage-0-to-later-stage acoustic predictability probe is also retained as a *separate diagnostic*, with no subtractive context regression activated.
- Save a JSON sidecar of train-only bank offset proposals; set its correction strength to 0 when the development screen fails. This sidecar is explicitly not a production voicebank setting, and **even a development-passing proposal is not automatically deployed**.

### CLI

```bash
python -m phonoweave.native_nuisance_v12 \
  --v9-results /path/to/frozen_v9_audio_results \
  --v10-results /path/to/frozen_v10_results \
  --output /path/to/NEW_v12_output
```

Or operate on user-supplied, independently held full WAVs:

```bash
python -m phonoweave.native_nuisance_v12 \
  --full-wav-zip /path/to/private_unmodified_recordings.zip \
  --output /path/to/NEW_v12_full_wav_output
```

Outputs: `summary.json` and `research_bank_offsets.json`. They contain bank metadata and source hashes, no raw recordings; review metadata before public disclosure. Running v12 does not edit the original archive.

## Actual retrospective LynneQing experiment

From 168 historical **OTO-curated** 600-ms audio excerpts: 503 unanchored acoustic regions, 23 script/clip families, 7 bank configurations, and 2 style families (`lu`, `mang`), all derived from one historical singer. Training/validation/audit family counts by stage: **14/5/4**. This is not independent-speaker validation, and group audits have already been examined in earlier prototype iterations, so no confirmatory generalization is claimed.

| Auto region index (not known phone position) | Validation raw top1 | Validation proposed top1 | Diagnostic λ | Validation distance-ratio change | Exploratory bank-identity null p | Development screen | Audit raw top1 | Audit proposed top1 | Audit matched/distractor ratio raw → proposed |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0 | .491667 | .533333 | 1.0 | +12.2% | .030303 | passes | .739583 | .739583 | .661239 → .528625 |
| 1 | .303571 | .312500 | .25 | +1.4% | .515152 | rejects | .302083 | .354167 | .948781 → .943114 |
| 2 | .350000 | .416667 | 1.0 | +8.9% | .030303 | passes | .333333 | .354167 | .979141 → .981770 |

**Interpretation:** The bank-offset model has modest validation improvements in spectral retrieval of the *same recording template* at indices 0 and 2. Index 1 fails the development screen, and exported effective strength is zero. At index 2 the retrospective audit top-1 rises slightly but matched-to-distractor contrast becomes **worse**, so performance does not consistently generalize even to historical held-out carrier families. At index 0 the audit distance ratio improves without top-1 improvement. Those outcomes do not justify removing style/pitch variation in synthesis. No automatic context acoustic subtraction passed v11's validation controls for stages 1 or 2; v12 does not turn it on.

The acoustic retrieval denominator is *matched carrier templates*; no reference phone identity, word transcription, perceptual similarity, synthesis A/B or human phone boundary annotation was used. Reported 49%–53%, etc. are not phone classification accuracy. Differences may reflect clip selection, temporal position, pitch, timbre, or recording conditions instead of linguistic structure.

## Independent tests and next hypothesis

Run `pytest -q tests` for v9–v12 module regression. The v12-specific tests cover frozen train-only provenance, source-family/PCM conflict rejection, matched bank permutation controls, exact within-bank distance preservation, positive/negative synthetic bank effects, unchanged validation decisions after audit changes, raw WAV extraction, empty-source abstention, sidecar non-deployment and non-destructive output.

**Next research priority:** independently recorded continuous WAVs across new contexts and singers. Evaluate whether a *selective* bank correction improves synthesis-relevant alignment or held-out perception. Preserve the raw representation and context-dependent realization information as separate channels rather than constructing a single 'corrected phone' without evidence.
