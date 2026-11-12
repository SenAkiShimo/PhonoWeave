# PhonoWeave v4.3 — first audible A/B for *existing* VC units

## Why this follows the actual v4.2 results

The user's real `cross_bank_proxy.json` + `cross_bank_review.csv` contain 51 phonemes, of which 50 have cross-pitch-bank proxy scores and `C:p` abstains. K=2 beats the K=1 spectral proxy on 26 of 50 measured phonemes. The largest best-K proxy changes are V:i (K=3), V:u (K=3), C:h (K=4), V:un (K=3), C:l (K=4), V:uan (K=2). **These are not real split decisions.** Four pitch banks of one singer are correlated and test-bank-centering is transductive. Many vowels have 1–2 original source templates. No invented acoustic split or new reclist is permitted.

An important separate question remains: **Does an already recorded VC transition actually add value compared with using existing CV units alone?** This runnable prototype renders real WAV snippets so an author can *listen* to one candidate at a time rather than relying on alias taxonomy.

## What A and B are

For each matched `previous V + next C` transition, three original OTO entries are needed: a **left CV** with the previous vowel, an **existing VC entry** for the tested transition, and a **right CV** with the target consonant and vowel. Three independent decoded PCM sources are required, so we don't compare an OTO cut to another cut of the exact same recording.

- The two arms use the **identical left-CV vowel excerpt** and **identical right-CV vowel excerpt**, same pitch bank, sample rate, splice method, target middle-segment duration, and 90-ms edge silence.
- **VC condition** inserts an excerpt from the actual existing VC unit between these flanks.
- **CV-only condition** instead inserts the consonant-region excerpt from the same right CV, omitting the dedicated VC transition.
- The mapping to `A` and `B` is stable and pseudorandomized per case. The listener-facing CSV does not expose it. The key is kept in `private_condition_key.json`—do not look at it until scoring.
- This keeps source budget/outer CV audio fixed but **does not equalize historical recording time invested**. It is a source-splice audition, **not** a native singing synthesis or a validated ablation with a complete renderer. A plain consonant fallback may itself be very poor; this is a limited, deliberately transparent first reference condition.

No existing VC aliases supply allophone subtype ground truth or determine how many subtypes to split. VC labels are used only to identify units already present, whose *utility* we are testing. No sound is removed, renamed or edited.

## Command using the actual local singer

```bash
cd ~/Downloads/PhonoWeave
git fetch origin
git switch research/vc-marginal-value-audio-ab-v4.3
source .venv/bin/activate
python -m phonoweave.vc_audio_ablation_v43 \
  "$HOME/Library/OpenUtau/Singers/LynneQing_CVVChn_Mang" \
  --max-pairs 12 \
  --output "$HOME/Downloads/phonoweave_vc_audio_v43"
```

Optional `--consonant sh` limits automatic matching to one consonant. Automatic matching is currently a **Mandarin-specific convenience adapter** using CV full syllable and existing `V C` OTO alias patterns. It is not appropriate for arbitrary languages and might find zero eligible pairs in some voicebanks. For any language use a user-authored **exact alias triplet manifest**, with explicit CV and VC references:

```json
{
  "schema": "phonoweave.vc_ab_cases.v1",
  "cases": [
    {
      "id": "case-1",
      "bank": "D4",
      "preceding_vowel": "a",
      "next_consonant": "sh",
      "left_cv_alias": "ma",
      "vc_alias": "a sh",
      "right_cv_alias": "sha"
    }
  ]
}
```

Replace illustrative bank/aliases with actual OTO identities; this file is **selection and provenance**, not the old split scheme as desired phonetic classes. Invoke with `--cases /path/to/exact_cases.json` in place of auto mode.

## Output and safe interpretation

- `blind_audio/pair_001_A.wav` and `pair_001_B.wav` etc.: short audio snippets for blind comparison
- `listening_sheet.csv`: pair identifier, context, A/B filenames, blank preference/notes columns
- `private_condition_key.json`: source OTO aliases/line numbers, WAV names, PCM SHA-256, condition identities and any exclusions

The two conditions are comparable in outer CV samples and approximate splice construction, but **do not isolate the real-world optimal resampler, phonotactic quality, pitch movement, and note boundaries**. Matching physical audio durations still leaves phonetic phase, stress and recording style differences. Blind listener preference for the VC arm under this simple construction would show **local audible splice value**, not global necessity of that recorded unit. Preference for CV-only would not prove the VC should be deleted.

The next experiment should use real score-aligned singing synthesis with a renderer and *matched total recording-budget* inventory policy, followed by listeners and confidence checks. For individual allophone splits, the same care is required: CV/OTO acoustic patterns are candidate evidence, not a label-based rationale to keep/multiply VC units.

## Tests

`python -m pytest -q tests/test_vc_audio_ablation_v43.py`

Synthetic tests cover WAV rendering from 3 separate OTO-located CV/VC recordings, same-duration A/B, blind key, no mutation, missing alias refusal, too-short VC rejection, source PCM overlap refusal. Synthetic listening tests do not support singer-specific efficacy.
