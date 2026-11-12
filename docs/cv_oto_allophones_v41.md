# PhonoWeave v4.1 — CV OTO-localized allophone discovery, without VC split supervision

## Research goal

For each author-described phoneme C or V, inspect all of its CV realizations in that singer's current voicebank and ask whether shared, two-way or three-way context-conditioned acoustic *models* improve source-template-held-out reconstruction. A context group is a candidate allophone hypothesis, **not** an approved recording unit. The number of groups is not copied from existing VC/CVVC aliases.

**OTO is useful evidence for sound location.** In CV samples, the algorithm uses the preexisting OTO offset and preutterance as approximate consonant span anchors; for the vowel, it uses the later region and respects the standard positive/negative cutoff semantics. **OTO is not used to tell the model how many VC units to retain or how to name/partition allophones.** Existing VC aliases are excluded from inference.

## Input modes

The default for the current Mandarin CV bank is an *opt-in Mandarin orthographic adapter*: full single-syllable CV tokens such as sha, shu, ma. The adapter maps the whole token to declared C and V symbols. It never treats a token pair such as "a sh" as a full CV; nor does the existing VC inventory prescribe a distinct class.

For other languages / singer orthographies use author CV lexicon JSON, not an IPA guess:

```json
{
  "schema": "phonoweave.author_cv_lexicon.v1",
  "alias_prefixes": ["D4- ", "G4- "],
  "cv": [
    {"cv": "sha", "consonant": "sh", "vowel": "a"},
    {"cv": "shu", "consonant": "sh", "vowel": "u"},
    {"cv": "ma", "consonant": "m", "vowel": "a"}
  ]
}
```

The original OTO alias is still read to locate a *whole CV* row corresponding to the author's CV dictionary; the *old VC split pattern* never supplies a target class or decision. Exact dictionary tokens are not reinterpreted. The temporary example lexicon is **not** a complete inventory.

## Command for the actual voicebank

```bash
cd ~/Downloads/PhonoWeave
git fetch origin
git switch research/cv-oto-allophone-v4.1
source .venv/bin/activate

python -m phonoweave.cv_oto_allophones_v41 \
  "$HOME/Library/OpenUtau/Singers/LynneQing_CVVChn_Mang" \
  --adapter mandarin \
  --output "$HOME/Downloads/phonoweave_cv_oto_v41"
```

The module is callable as `python -m ...`, independent of older CLI routing. Use a fresh output path outside the singer's bank. Other languages use `--adapter author --cv-lexicon /absolute/path/to/cv_lexicon.json`.

Outputs:

- `allophone_candidates.json`: original CV alias/WAV/OTO provenance; candidate C and V sound regions; within-phoneme candidate partitions for K=1/2/3; source-filename-template-held-out errors and reasons for abstention
- `allophone_review.csv`: human-readable per-C/per-V comparison and whether observed source coverage is sufficient

## Boundaries of what is actually demonstrated

- OTO timing is treated as approximate user-provided sound localization, **not as independently verified full phone boundaries**; negative cutoff means duration from offset, positive cutoff means trim from WAV end. A bad OTO region can cause rejection.
- Frequency-domain C/V profiles are only 3x12 spectral shapes; they can reflect coarticulation, the neighboring sound, source pitch, style and renderer artifacts.
- Source-family holdout prevents one filename template appearing in train and test across pitches, but the same singer's correlated renders remain correlated. A positive K1–K2 MSE does not prove a useful split or ideal category count.
- K=1/2/3 are exploratory candidates only; not assuming all phonemes have a fixed two-way division. More contexts may eventually demand K>3 or different representations. The number of partitions must be judged against acoustic and synthesis evidence, not old VC alias counts.
- **Existing VC marginal value has not yet been measured.** That follow-up must perform source-preserving, controlled same-phrase resynthesis with / without each pre-existing VC unit (and a fair substitution/budget control), not compare different VC alias tags. Nothing gets removed from the singer.
- No actual voicebank is present in the ChatGPT runtime. User must run this against the existing singer files and return the source-private report. No raw WAV is uploaded or committed.

## Tests

`python -m pytest -q tests/test_cv_oto_allophones_v41.py`

Synthetic tests cover real OTO-to-WAV acoustic extraction, excluding VC aliases, source-template holdout, user-defined multilingual sound labels, duplicated alias collapse, negative-cutoff vowel handling, and no source mutation.
