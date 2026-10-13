# Synthesis relevance v0.9 — paired phonetic finals and donor identifiability

## Hypothesis and limits

Compare a donor of the **identical target final** with a donor of the **prespecified rounded/plain counterpart**: `a/ua`, `ai/uai`, `an/uan`, `ang/uang`. Target, donor pitch subbank, physical-WAV isolation and duration tolerances are held fixed. This is an offline proxy, not real OpenUtau-rendered audio.

This pairing controls grossly unrelated final differences but **does not isolate the original /sh/ consonant**: the contrasting glide /u/ changes syllable-level acoustic context, and OTO preutterance remains a renderer timing parameter. We do not claim matched underlying nuclear identity across all pronunciations or universally generalizable causal coarticulation.

## Prespecified design

- Reuse v0.8 provisional core and fixed-right 48-dimensional log-spectral signatures from real WAV and OTO files.
- Fix four final pairs before evaluation. Do not search all finals for the most favorable result.
- A valid target must have one **different physical WAV** carrying the identical final and one **different physical WAV** carrying its paired alternate final, in the same pitch subbank. Select nearest OTO preutterance duration independently of spectral similarity, with a maximum 10 ms absolute deviation in each arm and a maximum 5 ms inter-arm deviation of mismatch.
- One donor is selected per arm; provide matched target-pair details and explicit counts of candidate unavailability.
- Compare cross-minus-same core reconstruction spectral cost (primary), fixed target-right boundary cost and RMS mismatch (secondary).
- Aggregate multiple targets within a physical target WAV before bootstrap and sign-flip statistics. Report donor reuse; WAV aggregation alone does **not** remove shared-donor dependence.
- Separately enumerate exact final pairs co-recorded in the **same physical WAV**. Such pairs are useful for descriptive context comparison but **cannot serve as held-out donor substitution** under physical-WAV isolation.

## Identifiability and expected sparse banks

Eight-bank v0.8 metadata already show that many finals are represented by only one distinct WAV per pitch subbank; strict same-final donor matching may therefore be unavailable in many banks. The correct status is `insufficient`, not a significant null or permission to pool unrelated finals. The current sample pool is not sufficient for strong inference about inventory necessity.

## Run

```bash
cd ~/Downloads/PhonoWeave
git fetch origin
git switch --track origin/research/synthesis-relevance-final-matched-v0.9
source .venv/bin/activate
python -m pytest -q
python -m phonoweave.synthesis_relevance_final_matched_cli \
  ~/.config/phonoweave/voicebanks.json \
  --base sh \
  --output ~/Downloads/phonoweave_sh_final_matched_v0.9.json
```

Output is JSON only; no audio generated, no listening task and no source voicebank modifications.
