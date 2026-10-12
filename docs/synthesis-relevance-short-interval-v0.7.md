# Synthesis relevance: short OTO-interval acoustic validation v0.7

## Research question

v0.6 found that excluded OTO candidates were overwhelmingly shorter than the previous **110 ms** minimum. In particular Cha Yue lost 144 candidate events on this gate. This pilot tests whether a conservative **40 ms eligibility floor** supports a larger recorded population while still providing informative **provisional OTO interval** features. It is not a new verified phonetic segmentation algorithm, synthesis test, or proof that plain and rounded UTAU units need separate recordings.

## Frozen signal measurements

- Do not change existing v0.5 25-ms/110-ms feature implementation.
- Inside OTO `[offset, offset+preutterance]`, leave up to 8 ms guard at either end and place three fixed 12-ms Hann windows at 15%, 50%, 85% of eligible center travel. Only OTO intervals at least 40 ms long are processed.
- Each frame returns COG (1.5–11 kHz), high-band fraction (>=5 kHz), and spectral flatness. Return nine time-major spectral coordinates. No neighboring WAV samples beyond OTO coordinates are accessed.
- Two alternate 12-ms layouts (early-biased 0/20/40% and late-biased 60/80/100%) act as contamination/window-sensitivity diagnostics. A strong late-only class association is *compatible with*, but not proof of, neighboring vowel or glide contamination.
- Source WAVs and OTO are only read. No sound, model audio, singer metadata, or production inventory is written.

## Evaluation

- Five-fold cross-validation with physical WAV grouping; per-fold feature standardization is fit only to training WAVs. Also retain the whole-WAV-label-flip conditional negative control, same-WAV paired sign-flip contrast, and leave-one-pitch-subbank-out results.
- Report valid events, unique WAVs, and short/long OTO duration distributions separately by context.
- To avoid a coverage confound, compare old 25-ms/110-ms and new 12-ms/40-ms features **on the same physically identified WAV/context cells** by averaging duplicate OTO examples per cell. Report sample count for the shared population; these are not exact event-level pairs if several alias events occur in one WAV/context.
- Do not compare raw 110 ms-only BA against full 40 ms+ BA as though a changed denominator were an algorithm gain.
- Windows are not independently reviewed as frication-only. Short windows can capture glide/vowel portions; all results remain **exploratory**. Different final distributions and OTO style may drive a classifier even after WAV grouping.

## Run

```bash
cd ~/Downloads/PhonoWeave
source .venv/bin/activate
python -m pytest -q
python -m phonoweave.synthesis_relevance_short_interval_cli \
  ~/Downloads/phonoweave_voicebanks.json \
  --base sh --permutations 199 \
  --output ~/Downloads/phonoweave_sh_short_interval_v0.7.json
```

The JSON includes all diagnostics and limitations. If a model cannot be evaluated because too few WAV groups contain both context classes, its status remains `insufficient` rather than inventing a numerical accuracy.
