# Architecture

PhonoWeave separates speaker analysis from recording-scheme generation.

## Pipeline

```text
recordings / voicebank
        |
        v
segmentation and feature extraction
        |
        v
speaker realization profile
        |
        v
contrast decision
        |
        v
synthesis unit inventory
        |
        v
scheme compiler
        |
        +--> reclist
        +--> alias map
        +--> oto suggestions
        +--> supplementary recordings
```

## Core layers

### Analysis

Extract comparable regions from recordings and measure context-dependent realizations.

### Contrast decision

Estimate whether an observed difference is stable enough and useful enough to justify a separate synthesis unit.

### Speaker profile

Store the result without tying it to a specific reclist format or alias convention.

### Scheme compiler

Convert a speaker profile into a target recording scheme. The compiler handles unit coverage, alias assignment, line length, repetition policy and supplementary recording generation.

## Language modules

Language modules provide expected phonological structure, candidate contrasts, legal contexts and scheme conventions. They do not replace acoustic analysis.

The Mandarin module is the first implementation. Other languages should reuse the same speaker-profile and compiler interfaces.
