# PhonoWeave · Mandarin whole-utterance forced-alignment bridge v0.1

## Scope

This is a working **source-preserving corpus → pretrained phone aligner → TextGrid import** pipeline. It is **not** a new `/sh/` or `/m/` detector. The same aligner can process every supported phone category. The Mandarin module supplies syllable/Pinyin transcription and model choice; the code does not infer IPA from OTO aliases.

### Real local run: LynneQing (DiffSinger output)

- Input: 7 pitch subbanks, 2 styles of one voicebank, 175 source WAVs, 13,539 OTO entries, 25 common WAV filename templates.
- **119** source WAVs have a candidate *whole-utterance syllable sequence*; these **119 actual full-length source WAV files** (534 MiB) were copied byte-exactly from the supplied ZIPs, paired with **119 `.lab`** documents representing **2,828 syllable tokens**.
- **56** source WAVs lack whole-utterance transcripts and were excluded from this stage, with explicit reasons and preserved source provenance. They are not permanently abandoned; later local-phone alignment must use verified local context and phone dictionaries.
- The 17 common template transcriptions are reconstructed from OTO **and remain unverified**. Staged real WAVs are labeled `UNVERIFIED_PILOT_ONLY` when `--allow-unverified-pilot` is used. In strict mode, **0** WAVs enter training/alignment until author or generation-score verification is recorded. The pipeline does not silently upgrade draft transcripts.
- **Model acoustic execution: NOT RUN in this tool environment.** `mfa` executable and model weights are not installed/cached, and the container network does not resolve the model host. **0 automatic whole-phoneme boundaries** and **0 measured boundary errors**. This is a genuine external-model dependency, not an excuse to substitute OTO times as model output.

## Prepare on a machine that has the 5 original ZIPs

The code expects the preceding whole-corpus preflight `output/` directory, including `source_wavs.csv`, `source_banks.csv`, and `transcript_templates_to_verify.json`, plus a `local_source_manifest.json` whose bank archive paths point to the five real ZIPs. The private local manifest, ZIPs, and any sound samples are not committed.

```bash
python -m phonoweave.whole_utterance_mfa_bridge prepare \
  --preflight /local/preflight/output \
  --templates /local/preflight/output/transcript_templates_to_verify.json \
  --manifest /local/preflight/local_source_manifest.json \
  --allow-unverified-pilot \
  --output /local/new_mfa_stage
```

**Important:** `--allow-unverified-pilot` explicitly authorizes **diagnostic** MFA alignment from unverified OTO-derived syllable sequences. It does *not* declare them correct. Leave it off for true model evaluation, and first fill template `verification` (e.g. `verified_against_source_score`) and an evidence reference in the input JSON. All original WAV/OTO data are preserved. Outputs are never overwritten.

## Run the real pretrained acoustic aligner

On Mac (e.g. with Miniforge's conda):

```bash
conda create -n phonoweave-mfa -c conda-forge python=3.11 montreal-forced-aligner
conda activate phonoweave-mfa
mfa --version
mfa model download acoustic mandarin_mfa

# MFA 3.4+ Hugging Face model format; confirm via `mfa align_hf --help`
python -m phonoweave.whole_utterance_mfa_bridge doctor
python -m phonoweave.whole_utterance_mfa_bridge align \
  --stage /local/new_mfa_stage \
  --output /local/new_mfa_textgrids \
  --mode hf
```

If your installed MFA exposes only the legacy CLI, use `--mode legacy` after downloading the *compatible* `mandarin_china_mfa` dictionary and legacy acoustic model, and verify the Pinyin dictionary covers the tokens. Check MFA's installation-version documentation before changing command signatures. **The pre-trained Mandarin model was developed for speech**, and its performance on DiffSinger-generated sung inventories must be measured, not assumed.

The adapter runs the real `mfa align_hf CORPUS mandarin_mfa OUTPUT --use_g2p --language unknown` command; it **does not** emulate MFA, train a fake model, or generate placeholder TextGrids. Once MFA succeeds, it writes a hash-bound execution receipt next to TextGrids.

## Import phone intervals into PhonoWeave

```bash
python -m phonoweave.whole_utterance_mfa_bridge import \
  --stage /local/new_mfa_stage \
  --textgrids /local/new_mfa_textgrids \
  --output /local/new_phone_interval_index
```

Produces `aligned_intervals.csv` (word and phone tiers, absolute time, source WAV identity), `alignment_quality_gate.csv` (missing tiers, word-count mismatch, overlaps), and `summary.json`. The import **refuses TextGrids without a successful MFA run receipt for the frozen corpus**. Transcript correctness and phone-boundary accuracy **remain unverified** until independent manual boundary references are evaluated. No alias or OTO is changed.

## Why this is different from previous VC studies

- Works at original-source-WAV scale, not per-alias or per-consonant.
- The acoustic model is **one multilingual-compatible alignment interface**; other languages replace acoustic/G2P models and transcript lexicons without new `/sh/`, `/m/`, etc. detector code.
- Original score/transcript is the one substantive bottleneck for 8 VC-only file templates. For the 17 other file templates there is a candidate sequence, but their accuracy still requires source-score verification.
- Author-specific nucleus symbols (`i`, `ir`, `i0`) stay as authored metadata; MFA's own phone symbols are **model outputs**, not automatically aliases or pronunciation truths. No split/merge policy follows from simply running MFA.

## Reproducibility checks

`pytest -q tests/test_whole_utterance_mfa_bridge.py`

The six regression tests use *synthetic* tiny WAV/OTO fixtures, not real Malay/Mandarin phone gold. Real local smoke checks completed: all 119 staged WAVs are present; sample of 7 bank/source waveforms exactly matches the original ZIP members; 56 excluded; no source mutation. Acoustically aligned real phone intervals remain **0** until pretrained model is installed and run.
