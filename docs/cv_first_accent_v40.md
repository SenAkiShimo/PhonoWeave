# PhonoWeave v4.0 — CV-first accent research input contract

The **only** input for this branch is author-labeled CV recordings and raw WAV audio. Existing `oto.ini`, alias splits, OTO offset/preutterance and Mandarin-specific CVVC definitions are **never read**. The system never guesses consonant/vowel symbols from the CV spelling: the author supplies all three text fields `cv`, `consonant`, `vowel`.

## Research question

For a given singer, how does an author's SAME declared consonant change acoustically across different vowels, and how does the SAME declared vowel change across consonants? Is any resulting conditional difference consistent and useful enough to warrant extra synthesis units when a future recording scheme is compiled?

Examples: author-labeled `sha (sh,a)`, `shi (sh,i)`, `shu (sh,u)`, `ma (m,a)`, `mi (m,i)`. These are input examples **only**; no standard phonological interpretation, fixed inventory, Mandarin language or number of splits is assumed.

## Manifest

Save `cv_manifest.json` in the directory holding original CV WAVs. Paths must be relative to this directory. Each WAV currently contains **one isolated CV token**. The implementation deliberately does NOT mine old OTO to cut long source WAVs or infer C/V labels; for long source recordings, a separate CV segmentation/manifest creation stage is still required.

```json
{
  "schema": "phonoweave.cv_first_manifest.v1",
  "recordings": [
    {"id":"sha-take1","cv":"sha","consonant":"sh","vowel":"a","wav":"audio/sha_01.wav"},
    {"id":"shi-take1","cv":"shi","consonant":"sh","vowel":"i","wav":"audio/shi_01.wav"},
    {"id":"ma-take1","cv":"ma","consonant":"m","vowel":"a","wav":"audio/ma_01.wav"}
  ]
}
```

No IPA, universal mapping or token-concatenation assumption. If a user writes an orthography where the CV label does not literally concatenate C+V, that is fine: the labels remain independent input fields. The WAVs must be dry uncompressed mono PCM16 with an isolated CV, between 120 ms and 12 s. Original source WAVs remain unchanged.

## Run

```bash
python -m phonoweave.cv_first_accent_v40 /path/to/cv_manifest.json \
  --output /outside/new_cv_screen
```

Outputs: `cv_acoustic_discovery.json` (source hashes, 12-band early/late spectral trajectories and an audio-only provisional acoustic change point), `cv_comparison.csv` (same-C/across-V and same-V/across-C profile comparisons). Repeated PCM identical recordings are rejected. At least two distinct WAVs per comparison cell are required even for a descriptive cross-condition distance; otherwise the comparison is unscored. No new recording list or alias is produced at this stage.

## Critical scope limit

The extracted **early** and **late** regions are not confirmed consonant and vowel segments. The algorithm uses waveform energy and spectral change *without OTO*, but that does not validate C-V transition boundaries. A stop burst, aspiration, vowel formant movement, pitch drift or timing difference can move the spectral-change candidate. Even a repeatable descriptive distance can reflect expected coarticulation rather than an accent-specific need for splitting.

This is an executable, correct-input **first acoustic screening**, NOT a learned accent-conditioned phoneme inventory or an optimal recorder. The next milestones are waveform-only segmentation with independent gold-boundary validation, balanced and genuinely source-independent repeats, phonetic context normalization, same-budget pooled-versus-conditional real singing synthesis, and *then* cost-aware reclist generation. Do not claim a model-derived split before those tests.

## Tests

`python -m pytest -q tests/test_cv_first_accent_v40.py`

Regression checks prohibit old OTO fields, prove label separation, avoid filename-based phone guesses, reject identical PCM, show low-replication abstention and verify that output is outside original audio.
