# PhonoWeave v3.4 — matched-rime accent audit

## Why the v3.3 ranking was insufficient

The user's real v3.3 report contains 10 context comparisons. The positive q (+0.04326) and x (+0.03904) comparisons use different rimes (qia/qiang/qiao vs que/quan/qu, and xia/xiang/xiao vs xue/xuan/xu). This is strongly confounded by following vowels and is not evidence for new q/x phoneme units. Four pitch banks do not count as four independent singers.

v3.4 produces same-PCM-source, same-written-rime counterpart pairs where the optional u medial is the main **orthographically represented** difference, e.g. sha/shua, shai/shuai, shan/shuan. The original source file can be the same long WAV, so microphone and session effects are less likely to explain pairwise contrasts. However, the u medial itself naturally conditions coarticulation; these are better listening/annotation candidates, **not** consonant-isolated or accent-validated comparisons.

## Run

```bash
cd ~/Downloads/PhonoWeave
git fetch origin
git switch feature/matched-rime-accent-audit-v3.4
source .venv/bin/activate
python -m phonoweave.accent_matched_rime_v34 \
  "$HOME/Library/OpenUtau/Singers/LynneQing_CVVChn_Mang" \
  --output "$HOME/Downloads/phonoweave_matched_rime_01"
```

Optional --base sh restricts to sh. Optional --listening-rimes-per-base 0 skips snippet creation. The output must be a new folder outside the original bank.

Files:
- matched_rime_audit.json: matched source identities and early/late spectral differences
- matched_rime_pairs.csv: pairwise diagnostic list for review
- original_source_excerpts/: original PCM snippets for manual inspection, **not** two synthesized systems or a blind A/B test

Supported conservative built-in spelling endings: a/ua, ai/uai, an/uan, ang/uang, o/uo. No new alias, sound or phone is generated. These conservative Mandarin hypotheses cannot replace an author-specified contextual alignment for other languages.

## Science and safety

Matching one original WAV cannot prove independent repetitions; multiple OTO lines from it are collapsed. The OTO-defined candidate area is not verified phone timing. The u medial may influence the consonant by normal anticipatory coarticulation, so a difference is not automatically a speaker-specific accent effect. Source snippets can contain private singer audio and must not be published without permission. Source WAV and OTO remain unmodified, while split/share and recording decisions remain UNRESOLVED. The next decisive test requires reviewed consonant regions, independent source-group holdouts, equal-time-budget rendered singing A/B and perceptual validation. For this module's tests run: python -m pytest -q tests/test_accent_matched_rime_v34.py .
