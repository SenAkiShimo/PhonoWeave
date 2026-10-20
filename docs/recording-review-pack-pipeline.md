# Recording review pack pipeline

`python -m phonoweave.review_pack_pipeline` consolidates five existing
read-only stages into one local command. It does not install phonemizer
aliases, derive new OTO timing, approve proposed split/merge decisions,
or modify source WAV files.

## Usage

For any local voicebank directory:

```bash
python -m phonoweave.review_pack_pipeline \
  --voicebank /path/to/local/voicebank \
  --budget 60 --repeat-target 3 \
  --output /path/to/new/review-pack
```

Alternatively, use a locally registered manifest and its opaque bank ID:

```bash
python -m phonoweave.review_pack_pipeline \
  --manifest ~/.config/phonoweave/voicebanks.json \
  --bank-id example_bank \
  --budget 60 --repeat-target 3 \
  --output ~/Downloads/phonoweave_review_pack_01
```

No source voicebank names or locally generated review artifacts need to be
committed to Git. Use a new output directory each time. The program refuses
an existing destination and refuses to write inside the source voicebank.
If any intermediate stage fails, the temporary pack is discarded.

## Contents

- `scheme/`: original alias/OTO evidence, unit inventory and review tasks;
  `alias_records.jsonl` preserves source strings without CSV formula escaping.
- `reference/`: observed, pitch-checked reference recording sequences.
- `budget/`: provisional speaker-context budget proposal and candidate reclist.
- `line/`: source-aligned internal context candidate and annotator/tuner guides.
- `directives/`: per-syllable annotation targets and tuner context checks.
- `review_pack_manifest.json`: stage summary and SHA-256 digests of key artifacts.

Review-only status persists across all outputs. The candidate reclist may
exclude recorded material or necessary transitions; it is **not** an approved
replacement for the source bank. New alias assignment, OTO timing, role
approval, and synthesis-quality validation remain separate engineering work.
