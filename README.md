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

## Structure

```text
phonoweave/       core analysis and compilation code
languages/        language-specific inventories and rules
schemas/          shared data formats
docs/             design notes and architecture
tests/            tests
```

## Status

Early development. Mandarin CVVC/CVVChinese-style voicebanks are the first target.
