# PhonoWeave

Speaker-adaptive phonetic realization analysis and recording-scheme generation for concatenative singing synthesis.

PhonoWeave analyzes how a specific speaker realizes context-dependent sounds and uses that information to derive a recording inventory suited to that voice.

The project is language-independent at the core. Mandarin is the first supported language module.

## Goals

- analyze existing voicebanks and their oto data
- detect stable context-dependent phonetic differences
- distinguish useful synthesis contrasts from ordinary coarticulation
- generate speaker-specific recording schemes
- suggest alias and oto changes for existing banks
- generate supplementary recording lists when coverage is incomplete
- support configurable reclist lengths and packing strategies

## Current scope

The first working path inspects existing Mandarin CVVC/CVVChinese-style voicebanks. It scans `oto.ini` files, checks referenced WAV files, and groups retroflex consonant observations by vowel context.

```bash
pip install -e .
phonoweave inspect /path/to/voicebank
phonoweave inspect /path/to/voicebank --json
```

Acoustic split/merge decisions are not enabled yet. The current inspector is intended to establish reliable recording coverage before acoustic analysis is added.

## Structure

```text
phonoweave/       core analysis and compilation code
languages/        language-specific inventories and rules
schemas/          shared data formats
docs/             design notes and architecture
tests/            tests
```

## Status

Early development. Mandarin is the first target, with the core kept independent of any one recording scheme.
