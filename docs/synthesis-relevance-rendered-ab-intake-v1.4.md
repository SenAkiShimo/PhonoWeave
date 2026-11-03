# Frozen S01/S02 pilot: external render intake and blinded listening v1.4

This branch **does not synthesize audio itself**. It provides the missing, tested
handoff to an external singing synthesis renderer, with immutable session
identity, audio hashes, source-only donor declarations and a blinded A/B package.

## Research workflow

1. Complete and audit the 48-take controlled /sh/ corpus (S01 train,
   S02 held-out). Run the v1.3 acoustic proxy first.
2. Create immutable plans outside the corpus:

   \`\`\`bash
   python -m phonoweave.synthesis_relevance_rendered_intake_cli plan \
     /path/pilot/manifest.json --proxy /outside/proxy.json \
     --output /outside/render_plan
   \`\`\`

3. Use your external **real singing renderer** to render exactly the 24
   held-out S02 score targets under two conditions for each final:
   - **pooled**: S01 six paired-final source takes only.
   - **split**: S01 three target-final source takes only.

   Both conditions must render the *same score*, phonemizer/resampler
   settings, target pitch, and master gain, with identical target duration.
   The S02 recording is the **reference for later evaluation**, never a donor,
   conditioning audio, or engine input.

   Place the 48 actual PCM16/24 rendered audio files at paths specified in
   \`render_plan.json\` (for example \`S02_shua_01/pooled.wav\` and
   \`S02_shua_01/split.wav\`). Fill \`render_receipt_template.json\` with
   exact donor take IDs used, engine+resampler version, shared score and
   configuration SHA-256, and actual SHA-256 per WAV.

4. Validate actual WAVs and create a separate blind participant package:

   \`\`\`bash
   python -m phonoweave.synthesis_relevance_rendered_intake_cli blind \
     --plan /outside/render_plan/render_plan.json \
     --receipt /outside/renderer_receipt.json \
     --renders /outside/renders --corpus /path/pilot \
     --output /outside/participant_package \
     --private-key /private/answer_key.json
   \`\`\`

   The command chooses a random blinding seed; 12 of the 24 trials assign
   pooled to A and 12 to B. Participants see only \`manifest.json\`,
   \`response_template.json\`, and anonymized A/B WAVs. Do **not** distribute
   the private key or renderer receipts to participants.

5. Collect independent response templates, setting \`listener_id\` and each
   trial's \`choice\` to \`A\`, \`B\`, \`tie\` or \`unusable\`; use \`unrated\` only
   for a skipped trial. Score using:

   \`\`\`bash
   python -m phonoweave.synthesis_relevance_blind_score_cli \
     --blind /outside/participant_package \
     --private-key /private/answer_key.json \
     --response /outside/listener01.json \
     --response /outside/listener02.json \
     --output /outside/descriptive_score.json
   \`\`\`

## Checks and limits

- Every task uses S02 as held-out **target** and has a separate six-take S01
  pooled pool and three-take S01 same-final pool. The plan re-audits the
  capture manifest and recomputes the frozen proxy to reject altered rankings.
- Receipt donor IDs must be nonempty subsets of the corresponding S01 pool,
  with no S02 donor ID. **This checks declarations, not actual renderer
  routing**. External renderer execution and settings are not independently
  inspected here.
- Every rendered WAV must match its receipt SHA-256 and be mono uncompressed
  PCM16/24 at 44.1/48/96 kHz, with identical frame counts and sample rates
  across the A/B pair. Byte-identical arms, tampered audio and audio identical
  to the reference are rejected. Bitwise distinct does not prove audibility.
- Audio is copied into a new non-overwriting directory, with SHA-256 checked
  again. Answers and source identities stay in an independently placed
  private key, separate from participant media.
- Scores are **descriptive listener votes only**: no p-values, MOS claims,
  cross-speaker generalization, rendered improvement or independence claim.
  There is no real captured or rendered dataset bundled in this repository.

Prior v0.3 temporary OpenUtau singers use OTO-derived provisional boundaries,
not this manually annotated controlled corpus. They are intentionally not
silently reused as this protocol's renderer.

Run \`pytest -q tests/test_synthesis_relevance_rendered_intake.py\` and
\`pytest -q\`. Both test suites use synthetic WAVs and cannot validate real
synthesis fidelity.
