# v1.1: Note-conditioned DiffSinger-source CVVC phrase previews

**Purpose:** move beyond the old 120ms VC→CV micro-splice to note-timed, pitch-controlled, continuous six-note phrase audio using the **original DiffSinger-generated source ZIPs**. Compare only different K=2 VC-sharing policies at identical unit budgets. This is an *engineering and listening-feasibility stage*, **not** a measured quality or split/merge superiority result.

## Architecture and scope

- The single author-confirmed LynneQing DiffSinger voicebank has Mang and Lu voice styles; 4 + 3 pitch subbanks (G3, D4, G4, D5 / C4, F4, B4).
- For each pair, the renderer extracts author-named onset-syllable CV (`li`, `shi`, `zi`), right-context VC (`i/ir/i0 → m/sh`) and corresponding next-syllable CV (`ma`, `sha`) from the **same style and pitch subbank**. Sources remain OTO-relative provisional intervals (a subset of Mang D4 boundaries was separately reviewed, but these new extracted windows are not certified across seven subbanks).
- Actual note pitch and time scaling uses the FFmpeg [`rubberband` filter](https://ffmpeg.org/ffmpeg-filters.html#rubberband), preserving formants, and each pair uses one fixed 12ms equal-power crossfade. Each note onset/duration is scheduled according to a frozen score. No source ZIP/OTO is overwritten. Source pitch is assumed from the subbank label, with independent F0 spot-check supplied separately.
- The renderer creates two **six-note** connected phrases per bank from six frozen two-syllable transitions: 7 subbanks × 2 phrases × 4 policies = 56 WAVs. Two phrases contain six VC targets total, one each. The score is fixed across all K2 arms and expressible in exported standard MIDI files with Pinyin lyric meta-events.
- **This is not an OpenUtau renderer**, `worldline` resampler, classic resampler, wavtool, continuous F0 editor or full DiffSinger rendering. It does not guarantee singing naturalness, correct Mandarin phonemizer handling or true phone boundaries.

## Budget and controls

| Policy | Kept VC classes per next consonant | VC donor inventory / subbank | Full selected unique source inventory |
| --- | --- | ---: | ---: |
| `k2_keep_i_ir` | `i`, `ir`; i0 uses i | 480ms | 2140ms |
| `k2_keep_i_i0` | `i`, `i0`; ir uses i | 480ms | 2140ms |
| `k2_keep_ir_i0` | `ir`, `i0`; i uses ir | 480ms | 2140ms |
| `k3_preserve_all_not_budget_matched` | all three classes | **720ms** | **2380ms** |

Unique source budget = VC donor duration plus three first-syllable CV fixtures (3×300ms) and two second-syllable CV fixtures (2×380ms), deduplicated across all rendered notes. K3 **must not** be used in the matched-budget winner comparison. No cross-style or cross-pitch source sharing occurs.

## Commands

```
# Input manifest (same schema as PR #70), with 7 subbank descriptors:
phonoweave diffsinger-note-phrase-ab \
  --manifest /private/lynneqing_sources.json \
  --score examples/lynneqing_frozen_note_score_v1.json \
  --output /private/new_preview_folder

# Optional, independent output directory for same-budget blind listening:
phonoweave diffsinger-note-phrase-blind \
  --renders /private/new_preview_folder \
  --bank mang_D4 --bank lu_C4 \
  --output /private/new_blind_package
```

The note renderer requires FFmpeg built with `--enable-librubberband`, and exits rather than silently substituting linear-time stretching. Unit tests use an explicitly labeled synthetic transform to test timing, budgeting and safety without relying on a CI machine's native DSP availability. The real audio run used FFmpeg Rubber Band.

## Frozen target examples

`phrase_a`: li ma → shi sha → zi ma (three pairs / six pitched notes)

`phrase_b`: zi sha → li sha → shi ma

Each note has exact semitone offset from its bank root pitch and a fixed duration in milliseconds; the phrase file contains no separately rendered DiffSinger teacher audio. The renderer generates MIDI files with note and Pinyin lyric events to assist future generation of matched independent reference vocals.

## Study gates and missing evidence

- The output is audible and genuinely note-controlled; that is the **only claim of completed rendering**.
- All reference WAVs would need to be generated independently by DiffSinger with the exact target score, same voice style, pitch contour and timing, avoiding reuse of ZIP source audio. There is no existing independent teacher WAV or test-reference score in this submission.
- A future rigorous test must freeze training source inventory, test syllable sequences, renderer and cost measurement, then use held-out teacher-rendered sequences (not reproductions of the same inventory snippets). Objective alignment/continuity and blinded preference tests will be needed to infer benefits.
- The current six-target pilot explicitly uses VC contexts present in the source; **held-out-syllable-sequence validity is false**. MIDI target scores are *requests for future independent teacher renders*, not reference waveforms.
- Different pitch subbanks are generated outputs of the same voicebank, not independent human repeat recordings or proof of multiple model checkpoints. Source OTO templates are highly synchronized; phonetic boundaries are not confirmed across all subbanks.

## Test coverage

`pytest -q tests/test_diffsinger_note_phrase_ab.py tests/test_diffsinger_phrase_listening.py`

Tests verify exact score grid, pitch root parsing, equal budgets, real PCM WAV writing (test shim), source non-overwrite, independent teacher absence, score tamper rejection, same-style donor use, optional real FFmpeg Rubber Band smoke, blind A/B checksum verification and K3 exclusion. No perceptual ratings are synthesized or inferred.
