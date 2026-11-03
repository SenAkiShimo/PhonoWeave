# v1.5 — executable S01-only reference splice rendering

This milestone genuinely produces 48 PCM16 WAVs and a source-hash-bound render
receipt, ready for the existing v1.4 blind intake. It is a **simple control
stimulus renderer**, not OpenUtau, neural/DDSP, a singing synthesizer, a
pitch-correction algorithm, or a legitimate estimate of high-quality singing
resampling. The external engine adapter remains a separate milestone.

## Run

Prepare a valid v1.4 render plan from the 48-take preregistered corpus and
v1.3 frozen S01/S02 acoustic proxy. Then:

\`\`\`bash
python -m phonoweave.synthesis_relevance_reference_renderer_cli \
  --plan /outside/render_plan/render_plan.json \
  --manifest /recordings/pilot/manifest.json \
  --corpus /recordings/pilot \
  --output /outside/reference_renders
\`\`\`

The new output directory contains:

- 24 \`S02_*/pooled.wav\` files
- 24 \`S02_*/split.wav\` files
- \`renderer_receipt.json\` compatible with v1.4 intake
- \`reference_render_metadata.json\` documenting limitations.

Blind and score normally with the v1.4 commands using
\`--renders /outside/reference_renders\` and
\`--receipt /outside/reference_renders/renderer_receipt.json\`.

## Signal construction

For each S02 target, select a *fixed* S01 same-final donor (take 01) as
the common vowel tail. Split uses that same-final donor's frication.
The reference "pooled" condition instead uses the **opposite final's**
S01 donor frication. Note: this is **cross-context donor substitution**,
not pooling/averaging, despite the external receipt's legacy \`pooled\` name.
It must NEVER be reported as the genuine pooled-model baseline.

Use human-labeled S01 onset and CV boundary, require at least 40 ms frication
and 150 ms vowel, time-remap the frication to 180 ms and shared vowel to
620 ms, overlap with an 8-ms equal-power crossfade, and save PCM16 mono 48kHz.
There is no melody or F0 transposition and no modern resampler engine.

This contrast is useful to check the pipeline mechanics, audio accounting,
blind packaging and qualitative splice behavior, but cannot measure whether
a *properly trained pooled* donor representation is inferior to an adaptive
one. Such scientific evaluation requires the real renderer.

## Data boundaries

The implementation checks all 48 source file hashes against the frozen plan,
but **only decodes S01 waveforms**. S02 files are never audio donors,
conditioning sources or shared right-hand materials. Rendered output is
published to a new directory atomically, never inside the corpus.
The existing v1.4 render receipt verifier checks every output's WAV format,
hash, pair length, donor identity declarations and nonidentical conditions.

Synthetic regression tests cover real WAV production, 48-arm verification,
direct blind-pack compatibility, anti-overwrite, source tampering, and a
monkeypatch guard that fails on attempts to decode S02 audio.
