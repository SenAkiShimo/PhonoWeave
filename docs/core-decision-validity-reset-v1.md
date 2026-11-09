# PhonoWeave · Core-decision feasibility reset (v1)

## What went wrong in the last research sequence

The project is **a speaker-adaptive decision and recording-scheme compiler**. The DiffSinger-source audio → Rubber Band preview → one external teacher comparison proves a runnable DSP test harness, **not** that PhonoWeave learns when to split or share a transition. Continuing to rank three hand-selected K2 policies on the same teacher song cannot validate a learned rule. Most recently, the 0.940 / 0.956 / 0.969 teacher spectral values were *descriptive* and sensitive to alignment; they are not an adopted split decision.

## Source census, all seven subbanks, OTO reads only

For **each** bank (Mang G3, D4, G4, D5; Lu C4, F4, B4), the number of **distinct parent WAVs** for the six focus aliases is identical:

| Alias | distinct source WAVs | available WAV |
| :--- | ---: | :--- |
| i → m | 2 | eriin.wav; vcsection2.wav |
| ir → m | 1 | vcsection2.wav |
| i0 → m | 1 | vcsection2.wav |
| i → sh | 1 | vcsection3.wav |
| ir → sh | 1 | ingi0ir.wav |
| i0 → sh | 1 | vcsection3.wav |

Thus **5 of 6** conditions per bank have only one matched source WAV. Seven pitch variants are not seven independent-score utterances, given the same DiffSinger voicebank and replicated timing templates. Some different target labels also share the **same parent WAV**, so a naive alias-level train/test split leaks source context.

## The only available focus-label control: i → m

The same original alias `i m` exists in `eriin.wav` and `vcsection2.wav`. We computed descriptive normalized 12-band log-power shape distances from 5×16ms short windows at OTO-anchor relative times −40, −20, 0, +20, +40ms; the comparators `ir m` and `i0 m` are both in `vcsection2.wav`. This is an **OTO-aligned proxy**, not an independently verified phoneme-boundary measurement.

Across seven subbanks, the within-label two-WAV distance ranges **0.490–0.945**; in Lu C4, it is **0.804**, *larger* than distances to `ir m` (0.431) and `i0 m` (0.676) drawn from the same WAV. That inversion shows how source/position/conditioning effects can exceed the apparently phonetic contrast; a universal distance cutoff cannot be inferred from these six aliases. `same_alias_negative_control.csv` contains the per-bank observations, not effect sizes or p-values.

## Next experiment: freeze the smallest new source evidence, not a new synthesized-human-recording study

The directory `scores/` contains two **genuinely different MIDI elicitation scores per voice style** for `mang_D4` and `lu_C4` (four MIDI total). Each score has six two-syllable phrases, covers all six original **alias labels** exactly once, and varies pitch contour, target ordering and duration. The MIDI notes and approximate target time slots are recorded in `source_score_targets.csv`; **MIDI note boundaries are not phonetic VC boundaries**. They must be extracted/reviewed from the new DiffSinger outputs later.

These are **new source-inventory elicitation renders**, *not* new teacher references and not automatically independent in the statistical sense. With two additional score-distinct parent WAVs per style and pitch, each target gains at least three **distinct source scores** for a first source-WAV-disjoint pilot. Whether the DiffSinger generator produces sufficiently distinct phonetic realizations must still be measured.

For study validity, freeze a **separate held-out teacher score** not used as a source sample; compare at **matched unique-source budget**, same rendering backend and voice style, and leave an entire source score/WAV out (not an individual alias window). Include an ordinary fixed-context baseline and an adaptable policy, predeclare the selection criterion without seeing held-out outcomes. Metrics must include join artifacts, F0/intelligibility, blinded listening or clear abstention. Pitch variants/style controls are *strata*, not independent human singers.

## Stop/go

**STOP:** No automatic approved split/share in `speaker_profile.yaml`, `synthesis_inventory.yaml`, or the recording scheme based on the current one-teacher test. Do not declare "keep i+i0 is optimal". No more additional copies of the same K2 preview score or spectrum rankings without new source-diverse evidence.

**GO:** Existing sound rendering and source verification can be reused unchanged. The *next* engineering loop should (a) import the new score-distinct sources, (b) review target-alias intervals without modifying original aliases, (c) freeze source-level train/development/held-out separation, (d) run policy vs matched-budget baseline, (e) expose an explicit `unresolved` decision unless all checks pass. New MIDI files here only prepare that experiment; they are not results.

The original five private ZIPs and `test2.wav` stay unmodified and are not bundled in this pack.
