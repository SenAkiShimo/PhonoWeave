# DiffSinger CVVC **micro-splice** pilot: actual PCM and equal source-budget policies

## What is implemented

Source-verified real audio from an author-confirmed **single** DiffSinger voicebank, with two styles (Mang and Lu) and pitch subbanks. Directly extract 120ms **OTO-relative** VC candidates and fixed 200ms `ma`/`sha` CV fixture from original ZIP WAVs. Produce audible WAVs using the **same** 12ms equal-power join and silence padding for all arms. No LLM-generated audio or fabricated listening scores.

For each style/pitch/right-context, the following strategies deliberately allocate **two 120ms VC donors** to the three target classes i/ir/i0; all three are *equal-budget selective-sharing policies*:

| Policy | i | ir | i0 | VC budget per m or sh |
| --- | --- | --- | --- | --- |
| `k2_keep_i_ir` | i | ir | i | 240ms |
| `k2_keep_i_i0` | i | i | i0 | 240ms |
| `k2_keep_ir_i0` | ir | ir | i0 | 240ms |
| `k3_preserve_all_not_budget_matched` | i | ir | i0 | 360ms |

The author's 3-unit fully preserved setup is a **higher-source-cost qualitative ceiling** only; it is **not** a matched-budget counterpart. The fixed CV (`ma`/`sha`) adds 200ms for each right context to every strategy, so training-source duration per two-right-context bank is 880ms for K2 and 1120ms for K3. The K2 budget is 4 VC donors × 120ms = 480ms; K3 is 6 × 120ms = 720ms. The repeated target renders reuse inventory units and are **not** additional budget.

### Run

```
phonoweave diffsinger-vc-microsplice-ab --manifest input.json --output /some/new/folder
```

Source manifest shape:

```json
{"schema":"phonoweave.diffsinger_vc_splice_input.v1",
 "source_provenance":"author_confirmed_diffsinger_direct",
 "voicebank_id":"LynneQing",
 "banks":[{"style":"mang","pitch":"D4","archive":"LynneQingD4.zip",
  "root":"LynneQingD4/","prefix":"D4"}]}
```

To keep original WAV archives private, none are uploaded to GitHub. The CLI fails on source ambiguities, malformed OTOs, missing WAVs, waveform clipping, and prior output directories; each output carries a source PCM hash, ZIP SHA, OTO line number, donor class/alias, budget ledger, and exact render settings.

## What the result **does not** test

**This is a reproducible, audible *micro-splice* from genuine DiffSinger bank audio, not full singing synthesis.** The pilot intentionally does **not** transpose pitch, stretch notes, run an OpenUtau or real resampler/wavtool, select units via PhonoWeave, or compare against a separately rendered, unseen DiffSinger teacher utterance. The source VC boundaries use fixed 120ms windows around OTO reference points, not manually validated consonant boundaries in every pitch/style. The 3 selective-sharing strategies are pre-registered exemplars, **not learned or optimized policies**.

Audibility can expose join failure and prove that real PCM travels end-to-end; it does **not** establish a quality winner or cost-utility tradeoff. Next milestone requires a real note-timed renderer with held-out teacher phrases and independent blinded listening.

## Local reproducibility

We executed the render using Mang D4 and Lu C4, producing 48 micro-splice WAVs (24 per style). The two styles were **not pooled** for donor selection. Source ZIP and audio were never modified. Six synthetic regression tests check budgets, PCM production, provenance rejection, missing OTO source, duplicate banks, and non-overwrite.
