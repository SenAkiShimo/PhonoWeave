# Synthesis Relevance v0.3: controlled OpenUtau A/B preparation

## Purpose

The corrected v0.2 hard-window boundary proxy found no stable pooled `sh` plain/rounded effect. A rendered test is required before any statement about singing quality. This utility generates a **new temporary UTAU-compatible singer** with explicit experimental aliases, enabling an OpenUtau operator to compare two controlled candidates.

## Construction

- Read the **saved v0.2 JSON** and original OTO/WAV audio. Select one recorded target event and two donor events from the same pitch subbank, matching and cross context, using **three distinct physical WAVs**.
- Extract at most 120 ms immediately preceding each event's provisional OTO `offset + preutterance`, and the exact same 600 ms segment immediately following the target's provisional boundary for both treatments.
- Join using a fixed 8 ms equal-power crossfade. All aliases have identical timing, sample rate, common gain, and target-right waveform. The third alias is a target-left/target-right control processed with the *same* stitching procedure. No original singer content is modified.
- Write `natural.wav`, `matched.wav`, `cross.wav`, `oto.ini` mapping `PW_N`, `PW_A`, `PW_B`, a temporary `character.txt`, and `experiment.json` with source provenance and output SHA-256. Refuse to overwrite an existing folder or write within the source singer.

The resulting samples have deliberately artificial boundaries. Both treatments contain the exact same *source* target-vowel samples after the splice; the donor consonant segments are different. **This is a hybrid experiment**, not naturally recorded CV units, and it does not determine whether OpenUtau's original CVVC OTO scheme ought to merge or split contexts.

## Run

```bash
python -m pytest -q
python -m phonoweave.synthesis_relevance_ab_staging_cli \
  "/Users/yang/Library/OpenUtau/Singers/LynneQing_CVVChn_Mang" \
  --pilot ~/Downloads/phonoweave_cv_substitution_pilot_v0.2.json \
  --context plain --subbank LynneQingD4 \
  --output-singer ~/Downloads/PhonoWeave_AB_D4
```

## OpenUtau procedure

1. Inspect the generated WAV waveforms around the splice (OTO preutterance time in `experiment.json`). If the mark does not correspond to the phonetic transition, flag this trial invalid instead of interpreting the results.
2. In OpenUtau, load the temporary singer (its folder may need to be copied into the OpenUtau singers directory), select the **Default** phonemizer (direct alias), and keep the renderer, resampler, note length, pitch, velocity, expressions and project identical.
3. Create one project with three separate test notes on three otherwise identical tracks, or three copies of the same project. For each note manually set the direct phonetic alias to `PW_N`, `PW_A`, or `PW_B` and **inspect the final phoneme/OTO selection** before export. If an alias is not found or another WAV is actually used, the trial is invalid.
4. Export dry mono audio from each version at the same sample rate, renderer settings and time range. Retain the project files, untouched raw exported WAVs and settings as reproducible provenance.
5. Loudness/peak check all three exports, and only after successful equivalence checks randomize the A/B presentation labels. Do not normalize A and B separately in ways that alter source gain ratios unless also reporting the change.
6. Run multiple independent targets/singers and separately measure splice naturalness and phoneme identity. An inability to hear a difference is a valid *inconclusive* result, not equivalence proof.

## Limitations

- This pilot does **not automatically render USTX** and does not claim to create a validated singing-voice file.
- OTO `preutterance` need not equal the true CV boundary. The correct splice point requires manual/phonetic validation.
- Same-context and cross-context donor consonants can differ in pitch, articulation and recorded phonetic contents.
- A hybrid splice may introduce artifacts absent in normal UTAU unit selection. A positive finding would establish a difference between these specific constructed stimuli, not automatically validate an entire voicebank reclist.
