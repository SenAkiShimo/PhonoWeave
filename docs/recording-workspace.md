# PhonoWeave Recording Workspace v0.1

An **offline, browser-based review workspace** for an already-generated,
checksummed PhonoWeave recording review pack. No external web server or
JavaScript framework is required. The output is one self-contained HTML file.
It does **not** turn a pilot recording plan into a production voicebank.

## Generate and open

```bash
cd ~/Downloads/PhonoWeave
source .venv/bin/activate
python -m phonoweave.recording_workspace \
  --pack ~/Downloads/phonoweave_review_pack_01 \
  --output ~/Downloads/phonoweave_workspace_01.html \
  --open
```

The HTML opens in the default browser. It contains the local project's review
data inline (including local original alias and WAV/OTO references); **do not
publish this generated HTML or commit it to a public repository**. The source
code, tests and this documentation do not contain third-party voicebank data.

## What the workspace shows

- **Project overview:** number of reference recording lines, budget-selected
  lines, syllable/pitch inspection cells and original OTO-linked candidates.
- **Recording scheme:** search recording lines, switch pitch layers, click
  syllable positions, and inspect the annotator/tuner directives and
  candidate original alias/OTO provenance.
- **Context units:** groups of source-backed internal context candidates,
  with original references and conservative review status.
- **Staff review desk:** review-plan items, source evidence, reviewer notes,
  and non-authoritative check progress.

Review notes live in the browser session only until explicitly exported to a
JSON file. You can re-import them into the **same** pack; the source manifest
SHA-256 prevents accidentally carrying them over to an unrelated compilation.
These notes never constitute an accepted split, new alias, recorded acoustic
judgment, measured OTO timing, or phonemizer routing approval.

## Integrity and privacy

- The source review pack must match all required manifest SHA-256 hashes.
- Cross-stage recording line/pitch/syllable identities are checked.
- Original WAV/OTO files and review-pack outputs are read-only.
- The HTML contains no external scripts, external fonts, APIs or telemetry.
- Output must be outside the source pack and will not overwrite an existing file.
- New alias naming, phonemizer routing and production OTO generation remain
  disabled. The report only presents existing recorded observations and
  provisional candidate decisions.

This first version opens an **existing review pack**, not a raw WAV folder.
To refresh an analysis, first run `python -m phonoweave.review_pack_pipeline`
with an appropriate local source and create a new workspace from that output.
