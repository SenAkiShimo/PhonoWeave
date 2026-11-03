# PhonoWeave language-neutral template import: first milestone

This milestone adds a **lossless text recording-list importer**, not a phoneme
recognizer or a finished recording-scheme compiler. It does not change the
existing Mandarin voicebank analysis.

## Quick start

```bash
phonoweave inspect-reclist /path/to/reclist.txt
phonoweave inspect-reclist /path/to/reclist.txt --separator underscore --json
phonoweave inspect-reclist /path/to/reclist.txt --encoding cp932
```

- Supports UTF-8 and legacy CP932 input; explicit encoding override available.
- Preserves full decoded source text, line endings, BOM, blank/comment lines
  and original line numbers.
- Tokenization is **explicit**: whitespace (default) or underscore.
- Tokens are literal strings; no assumption is made about syllable boundaries,
  language, IPA, or UTAU/OpenUtau alias mapping.
- Reports counts only; it does not yet export personalized lists.

## Next incremental steps

1. Add fixture-backed readers for representative community CV, VCV, CVVC and
   VCCV recording lists. Avoid assigning phonological meaning until the
   template convention is identified.
2. Introduce a language-pack interface with explicit orthography-to-phoneme
   mapping, allowed phonotactics, and contextual contrasts.
3. Connect language-pack analysis to existing speaker-profile decisions,
   tracking unknown or low-confidence items instead of silently guessing.
4. Introduce a recording-unit graph, editable split/merge recommendations,
   and a deterministic round-trip compiler back into supported formats.
5. Validate with held-out speakers and real resynthesis before treating
   acoustic proxy measures as evidence of perceptual improvements.

## Validation

```bash
pytest -q tests/test_reclist.py
pytest -q
```

Avoid importing unlicensed third-party recordings into this repository.
