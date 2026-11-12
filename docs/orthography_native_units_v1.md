# User-native acoustic units v1 — PhonoWeave's IPA-independent analysis path

## What this fixes

PhonoWeave's source of truth is the **user's language-specific romanization/orthography and voicebank aliases**, not an IPA inventory or MFA phone IDs. An internal unit ID is an opaque string supplied by the user or *proposed for review*. For instance, `i`, `ir`, `i0`, `i2` need no phonetic standard mapping. The separate `phonetic_transition_graph.py` already accepts explicitly authored unit labels, roles, and spelling; this module supplies **acoustic evidence** to that interface rather than replacing the user's mapping.

### Interface

`phonoweave discover-user-units --manifest manifest.json --output results`

The manifest specifies `language_id`, `user_base_symbol`, `user_realizations`, contextual `observations`, and source bank ZIPs. Each observation independently supplies:

- `user_symbol`: opaque name, which may be Latin, Chinese, Unicode, or a number-suffixed name;
- `context_id`: user-chosen context, not a hard-coded phoneme type;
- `source_wav` and `oto_alias_body`: exact source WAV and its **original** alias body, even when different from the `user_symbol`;
- optional `oto_line`: explicit disambiguation where there are duplicate OTO aliases.

A bank's `prefix` (such as D4) is only used to reconstruct its existing OTO source key. OTO-derived `anchor = offset + (preutterance+overlap)/2` locates a *provisional* comparison region; it is **not a phoneme onset or a gold boundary**. The core window is specified in `oto_relative_ms` and can vary across studies. The original WAV, OTO, and alias are never modified.

The pilot uses a 70ms pre-transition window, 25ms frame and 10ms hop, common 16kHz analysis rate, 30 mel bands, 12 normalized cepstral coefficients excluding overall gain. Features may be affected by pitch/style, timing or parent-WAV confounds. A user can supply one seed name to explore unlabeled subdivisions; the implementation does not mint an approved suffix.

### Distinguishing unit identities from model output

- Audio analysis uses no IPA table, G2P, phoneme regex, or MFA model.
- **Known-label transfer**: train on one following context and test on another; leave each pitch out within its style; train on one style and test the other (both from the same confirmed voicebank).
- **Blind clustering**: Ward clustering sees only the acoustic vectors and their **bank identities**, subtracting the unlabeled per-bank mean to control pitch/style common shifts. User labels are used **after** fitting for a post-hoc audit. Both raw and bank-centered alternatives are reported.
- **Unresolved class count**: silhouette is calculated for K=2..6, and a 3-cluster fit is *not* evidence that K=3 is automatically optimal. Normalizing a batch is transductive and requires multiple units per bank; a lone audio clip cannot be classified this way without adaptation.
- No automatic overwrite, no alias reindexing, no split or merge approval, and no claims about full phone boundaries. The `review_candidates.json` is intentionally *not* a production pronunciation lexicon.

## Real LynneQing pilot (author's one DiffSinger voicebank, two styles, 7 subbanks)

42 source-linked windows (`i`/`ir`/`i0` × following `m`/`sh` × seven subbanks), zero read/extraction abstentions.

| Test | Correct | Total |
| --- | ---: | ---: |
| Cross-following-consonant held-context classification | 42 | 42 |
| Whole-pitch holdout *within each voice style* | 42 | 42 |
| Whole-voice-style holdout (within same LynneQing model bank) | 39 | 42 |
| Forced K=3, raw unnormalized clustering, posthoc best alignment | 24 | 42 |
| Forced K=3, blind per-bank mean centered clustering, posthoc | 42 | 42 |

With per-bank centering, unsupervised silhouette scores are **K=2: 0.48276, K=3: 0.47708**. The best K is **2** by the predeclared single silhouette metric, with a very small gap. Do not report that the model *automatically discovered* exactly three phoneme classes; this pilot only supports that three **author-named** categories are *acoustically distinguishable* when tested. Additional K-selection evidence is needed to propose new labels.

Sensitivity to the **source-window choice**:

| Pre-transition window, relative to OTO anchor | Cross-context | Leave pitch out | Leave style out |
| --- | ---: | ---: | ---: |
| -150 to -80 ms | 42/42 | 41/42 | 40/42 |
| -125 to -55 ms (primary) | 42/42 | 42/42 | 39/42 |
| -105 to -35 ms | 41/42 | 40/42 | 38/42 |

These are model-rendered **non-independent voicebank samples** and do not measure within-label repeated takes, natural human phonetics, precise phone boundaries, perceptual distinctiveness or downstream split/share efficacy. The three candidate labels were already manually established, and the extracted signal may include coarticulation. No new user alias is recommended automatically.

### Acceptance gates for advancing beyond the pilot

1. For any new user language/orthography, preserve spelling, source alias and custom unit naming as separate versioned records.
2. Demonstrate stable acoustically meaningful grouping across context, bank and genuinely independent source/session repeats; candidate number of clusters must not be fixed to a known answer.
3. Validate real segmentation boundaries or provide conservative stable cores plus uncertainty intervals; a chosen OTO-relative window is not a boundary.
4. Only after matched source-budget concatenative renders and actual listening evidence should the recording-scheme/`speaker_profile` compiler treat a new split as approved. Otherwise **unresolved** is the correct output.

## Reproducibility and confidentiality

The repository contains only this source module, synthetic tests, an example manifest with placeholder ZIP paths, and the descriptive measured summary. No private voicebank ZIP or user audio is committed to GitHub. A separate private package supplies the source-linked 42-row sample index and measured per-sample features. Python >=3.11, NumPy and SciPy are used; no Chinese/G2P model or IPA dependency.
