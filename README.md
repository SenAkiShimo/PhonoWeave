# PhonoWeave

Speaker-adaptive phonetic realization analysis and recording-scheme generation for concatenative singing synthesis.

PhonoWeave analyzes how a specific speaker realizes context-dependent sounds, tests whether those differences are useful enough to justify separate synthesis units, and compiles the result into a speaker-specific synthesis inventory.

The core is intended to remain language-independent. Mandarin is the first supported language module.

## Current pipeline

```text
voicebank / recordings
        |
        v
voicebank inspection and context analysis
        |
        v
speaker realization decisions
        |
        v
speaker_profile.yaml
        |
        v
synthesis_inventory.yaml
```

The analysis layer and the synthesis inventory are deliberately separate. Internal realization IDs such as `sh_rounded` or `x_front_unrounded` are neutral analysis/compiler identifiers; UTAU/OpenUtau alias spelling is a later compilation step.

## Install

PhonoWeave currently targets Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Main commands

Inspect an existing voicebank:

```bash
phonoweave inspect /path/to/voicebank
```

Run the current speaker-adaptive inventory analysis:

```bash
phonoweave analyze-voicebank /path/to/voicebank
```

Build a neutral speaker realization profile:

```bash
phonoweave build-profile /path/to/voicebank -o speaker_profile.yaml
```

Compile the profile into synthesis units:

```bash
phonoweave build-synthesis-inventory /path/to/voicebank -o synthesis_inventory.yaml
```

Lower-level diagnostic commands remain available for supported Mandarin contrasts:

```bash
phonoweave analyze /path/to/voicebank --base sh
phonoweave analyze /path/to/voicebank --base s
phonoweave analyze /path/to/voicebank --base x

phonoweave analyze-affricate /path/to/voicebank --base zh
phonoweave analyze-affricate /path/to/voicebank --base ch
phonoweave analyze-affricate /path/to/voicebank --base z
phonoweave analyze-affricate /path/to/voicebank --base c
phonoweave analyze-affricate /path/to/voicebank --base j
phonoweave analyze-affricate /path/to/voicebank --base q

phonoweave analyze-rhotic /path/to/voicebank
phonoweave splice-test /path/to/voicebank --base sh
```

## Design principles

- Acoustic difference alone does not imply a separate synthesis unit.
- Split/merge decisions should be speaker-specific and context-aware.
- Failure to find significant evidence is not treated as proof of acoustic equivalence.
- Speaker profiles are neutral with respect to final alias conventions.
- Existing voicebanks are analyzed non-destructively.

## Current Mandarin coverage

The v0.0.1 core supports the current contrast-analysis path for:

- fricatives: `sh`, `s`, `x`
- affricates: `zh`, `ch`, `z`, `c`, `j`, `q`
- rhotic `r` with plain/front/rounded realization analysis

The current synthesis-relevance proxy is exploratory and does not replace perceptual validation.

## Project structure

```text
phonoweave/       core analysis and compilation code
languages/        language-specific inventories and rules
schemas/          shared data formats
docs/             design notes and architecture
tests/            tests
```

## Status

v0.0.1 is the first working core analysis milestone: existing Mandarin voicebanks can be inspected, analyzed, converted to a neutral speaker profile, and compiled into a synthesis-unit inventory. Alias compilation, adaptive OTO suggestions, and supplementary reclist generation are planned as later stages.
