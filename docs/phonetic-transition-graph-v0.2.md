# Explicit pronunciation transition graph v0.2

The module is intentionally speech-domain **generic**. The input lexicon
describes phones and nucleus/consonant roles; it does not assume Pinyin, Latin,
IPA, Russian Cyrillic, German orthography, OpenUtau aliases, or singing.

## Purpose

Capture a failure of literal spelling baselines to represent *distinct
pronunciation-defined transitions*, including across syllable boundaries.
This is a **symbolic coverage** baseline and **not** a new phonological
discovery, acoustic split classifier or synthesis experiment.

Alias/source tokens are retained intact. Every pronunciation must be supplied
explicitly in the JSON lexicon. A missing token always fails closed.

## Example

Create a plain UTF-8 reclist \`reclist.txt\`:

\`\`\`text
shi ma
li ma
\`\`\`

Provide \`lexicon.json\`:

\`\`\`json
{
  "schema": "phonoweave.pronunciation_mapping.v0.1",
  "language": "zh-CN-manually-reviewed-demo",
  "entries": {
    "shi": [
      {"symbol": "ʂ", "role": "C", "grapheme": "sh"},
      {"symbol": "ɻ̩", "role": "V", "grapheme": "i"}
    ],
    "li": [
      {"symbol": "l", "role": "C", "grapheme": "l"},
      {"symbol": "i", "role": "V", "grapheme": "i"}
    ],
    "ma": [
      {"symbol": "m", "role": "C", "grapheme": "m"},
      {"symbol": "a", "role": "V", "grapheme": "a"}
    ]
  }
}
\`\`\`

These example phone symbols reflect one Mandarin phonetic analysis, not
a canonical universal transcription. \`role="V"\` designates a **syllabic
nucleus** (including a syllabic approximant or consonant), not a claim that
all such phones are ordinary vowels.

Run:

\`\`\`bash
phonoweave plan-phonetic-transitions reclist.txt \
  --lexicon lexicon.json --budget 2 --output comparison.json
\`\`\`

## Graph semantics

Each token represents one syllable in this milestone. Each phone has an
explicit C/V role, independent of the spelling fragment (grapheme).
The graph contains:
- phone types;
- ordered within-syllable adjacent phone types;
- ordered *cross-syllable* adjacent phone types.

Consequently, CV, VC, VV and CC patterns are all representable inside
a syllable, across syllables, or both. No relation beyond immediate adjacency
(e.g. nonlocal consonant interactions or prosodic phrase effects) is modeled.
A lexicon entry must have a syllabic nucleus. Existing author alias names
are never rewritten.

The report exposes divergent phonetic realizations of an identical grapheme
as **lexicon-declared ambiguities**; it does not infer the relationship
by acoustic measurement or choose a proper alias for the author.

## Recording-budget comparison

With identical candidate reclist lines and a common maximum line count:
- Phonetic policy: greedy weighted novel phone/within/across transition types.
- Literal policy: existing token + adjacent literal alias pair greedy baseline.
- Report pairs the actual **common prefix at exactly k lines** for both
  policies, not unequal final totals.

Weights are fixed in v0.2 for this CLI (phone=1, within=2, across=3).
All are researcher-specified proxies. Recorded durations, pitch coverage,
phonetic equivalence, acoustic variations and actual synthesis quality
are **not** measured. Both approaches can stop early when coverage saturates.

## Next scientific steps

1. Freeze expert-verified language packs and cross-word/syllable metadata.
2. Attach observed phone/transition features and speaker-specific confidence.
3. Add held-out recording sessions and measured durations.
4. Compare against lexical, G2P/phoneme-aware, and context-sensitive
   baselines, including ablation of acoustic adaptation.
5. Show actual synthesis or intelligibility improvements at matched costs.

A success on this module alone is a spelling-aware coverage demo, not a
research result establishing speaker-adaptive improvements.

Tests:
\`\`\`bash
pytest -q tests/test_phonetic_transition_graph.py
\`\`\`
