# Source-linked phone-core segmentation — v0.1

## Aim

This milestone **cuts actual phone-like source audio regions** from immutable voicebank WAVs and maps those regions back to original OTO rows, archive hashes and absolute source-WAV timestamps. It is segmentation research, not phonetic unit sharing, reference singing, a resampler or a new recording scheme.

The author supplied a source-hash-checked human transition-band review for six Mang D4 contexts: `i/ir/i0 → m/sh`. The alias semantics (`i` nucleus of `li`, `ir` nucleus in `shi/ri`, `i0` nucleus in `zi/ci`) are *author-confirmed for this bank only*, not universal phonetic symbols. **Human transition bands are not exact point phone boundaries.** An automatically observed spectral rise is an acoustic cue, not a certified phone onset.

## What actually ran (author-provided seven subbanks)

- `42` VC transition candidates: the six human-review-selected contexts × Mang G3/D4/G4/D5 and Lu C4/F4/B4. All 42 yield **actual** source-PCM 600ms context clips, conservative 40ms left nucleus candidates, the uncertain transition band as audio and right consonant core candidates. Each has a Praat `phones.TextGrid` with **sparse phone cores and uncertain transitions**, not a fabricated full-phone alignment.
- `6/42` transition bands are author reviewed (Mang D4) and **their source clips were reproduced exactly, byte for byte**, from the original original WAVs in the ZIP the author reviewed. `36/42` are unreviewed transferred or acoustic-shift candidates. `/m/` has no validated nasal onset detector and is explicitly tagged as transferred, not detected.
- Independently scans all OTO aliases with the right side `sh`: `24` left contexts × `7` subbanks = **168 acoustic /sh/-like core candidates**, each an actual 40ms extract and 600ms source context. Each of the 168 passes a persistent high-band frication screen. Parent WAV reuse is explicit; these **are not 168 independent takes** (15 parent WAVs per subbank in this corpus).
- `12` synthetic regressions (with a positive source-linked end-to-end test), plus a real seven-bank run; no sources or private audio are committed to GitHub. ZIP SHA-256, OTO SHA-256 and original human review-pack excerpt-byte identities are verified.

## Reproduce

```bash
phonoweave segment-phone-cores \
  --manifest /private/bank_manifest.json \
  --review /private/review_annotations_completed_v07.csv \
  --review-pack /private/LynneQingD4_VC_Interactive_Review_v07.zip \
  --output /private/new_phone_segmentation
```

The manifest follows `phonoweave.multisubbank_input.v1`: each bank specifies `style`, `pitch`, `archive`, `root` inside the ZIP and source alias `prefix`. One bank must have ID `mang_D4` because it is the only one with verified review-package provenance in this pilot. No source file is modified; the output folder must not already exist.

## Outputs

`banks/<style_pitch>/<left>_to_<right>/`:
- `excerpt_600ms.wav`: bit-preserving source context, reconstructed from original mono PCM16 samples;
- `left_nucleus_core.wav`: conservative 40ms candidate **core**, not an entire nucleus;
- `uncertain_transition.wav`: full human-review or provisional acoustic interval;
- `right_consonant_core.wav`: conservative 40ms candidate **core**;
- `phones.TextGrid`: sparse cores and uncertain transition on a local 0–600ms timeline;
- `provenance.json`: absolute source times, original OTO alias and row, hashes, review status.

`all_sh_phone_cores/<style_pitch>/<left>_to_sh_OTO<line>/`: source context and 40ms frication core for all locally sound-supported `... sh` alias rows, with per-target and whole-corpus CSV. `segment_index.csv` and `segmentation_report.json` document all six target classes, all seven pitch subbanks, including abstentions.

## Scientific boundary / remaining work

- The current system extracts **phone cores** and uncertain transitions, not full phone start/end boundaries. `whole_phone_boundaries_confirmed=0`. Do not import these timestamps as a finalized UTAU OTO, a phone-aligned training target or a validated full forced alignment.
- A high-band rise helps locate a fricative acoustic event; it cannot detect nasal `/m/` boundaries. Those require separate cues (nasal murmur, formant anti-resonance, voicing/energy and articulatory/linguistic context) and additional human reference labels.
- No claim about the left phone identity is made for other `... sh` contexts outside the six author-confirmed `i/ir/i0` classes. No generalized IPA, cross-language aligner or multilingual phone segmentation yet.
- Next step for **actual phoneme segmentation**: obtain a small held-out set of **precisely defined phone-onset / offset annotations**; report onset errors against that, implement separate nasal and fricative evidence models, then extend to all context classes without promoting transferred OTO timestamps to gold boundaries.

**Never** use a successful cut or stable /sh/ acoustic core as evidence that PhonoWeave should split or share a phone for synthesis. That separate decision requires an independent effect test.
