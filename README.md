# PhonoWeave

[![tests](https://github.com/SenAkiShimo/PhonoWeave/actions/workflows/tests.yml/badge.svg)](https://github.com/SenAkiShimo/PhonoWeave/actions/workflows/tests.yml)

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

Run the whole chain in one pass. The speaker analysis is computed once and the command writes the synthesis inventory, the main reclist and an audit of the voicebank's own entries into one directory. Unresolved decisions are treated as one unit per onset and need no recording. The optional supplementary diagnostic reclist and evidence-gap report are written only with `--with-supplement`:

```bash
phonoweave plan /path/to/voicebank -o plan_output
phonoweave plan /path/to/voicebank -o plan_output --with-supplement
```

Audit the entries an existing voicebank already has (plain CV, initial CV, VC, end) and list what is missing in some pitch layers but present in others:

```bash
phonoweave audit-bank /path/to/voicebank
```

Compare the onset units chosen from the acoustic analysis with the context variants the voicebank's own VC labels already distinguish (plain, `y` before i-series finals, `w` before u-series finals):

```bash
phonoweave compare-bank /path/to/voicebank
```

`plan` writes the same comparison to `bank_concordance.txt`.

Generate a continuous recording list from the speaker's synthesis units:

```bash
phonoweave build-reclist /path/to/voicebank -o reclist.txt
phonoweave build-reclist /path/to/voicebank --inventory synthesis_inventory.yaml -o reclist.txt
```

`--finals-from-bank` makes the VC requirements follow the final classes the voicebank itself uses (the WAV file grouping of its CV entries) instead of one VC final per syllable final; the mapping is written to the report as `final_classes`. Without `--inventory` the speaker analysis is run first. The command writes the recording list and a `reclist.report.json` file with requirement counts, the line-count lower bound and a one-unit-per-onset baseline for comparison. See `docs/reclist-compiler.md`.

List the onsets whose decision is still unresolved, what evidence is missing, and a short supplementary diagnostic reclist for them:

```bash
phonoweave evidence-gaps /path/to/voicebank --write-reclist supplement_reclist.txt
```

Each supplementary line is meant to be recorded once at every pitch layer already present in the voicebank. After recording, re-run the analysis once; unresolved decisions stay frozen after that round.

Transition consistency between vowel contexts can be inspected without listening:

```bash
phonoweave transition-report /path/to/voicebank --base m n l r --json transition.json
```

For each onset, the report extracts a pitch-independent spectral envelope around the consonant-vowel boundary, removes the vowel by projecting each sample onto the line between its own consonant and vowel states, and compares the remaining path at 25%, 50% and 75% progress between context families, within each pitch layer. It is exploratory and does not change any inventory decision.

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
phonoweave splice-test /path/to/voicebank --base zh
```

## Experimental live calibration

The repository also contains an experimental calibration workflow for elicited recordings, separate from the existing voicebank analysis and synthesis-inventory compiler.

- Record a calibration session with `phonoweave-gui`.
- Annotate consonant onset and consonant–vowel (CV) boundary independently using `python -m phonoweave.calibration_manual_labeler /path/to/session`. The labeler supports sessions containing only a subset of recorded prompts, preserves existing annotation files, and displays synchronized waveform/spectrogram views.
- Evaluate paired onset/CV candidates and experimental noise-compensated onset refinements using the research CLIs under `phonoweave/calibration_*`. Development diagnostics and manual reference scores are not inputs to the default voicebank inventory compiler.
- Compare a new, independently recorded session with the frozen selective-onset procedure using `python -m phonoweave.calibration_independent_session_cli /path/to/new_session --development-session /path/to/development_session --device phone --environment quiet`. Device and environment metadata must reflect the actual capture conditions.

**Validation boundary:** The evaluated paired-fricative detector and selective backtrace were developed primarily for elicited Mandarin `x`/`sh` targets. Reported localization results are from limited development and independent-session recordings, not broad speaker, device, or environmental validation. The experimental calibrator is not yet connected to the production speaker-profile or synthesis-inventory decision rules. Do not interpret its exploratory timing outputs as a validated automatic recording scheme.

Implementation and study limitations are documented in `docs/calibration-*.md`.

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

The current synthesis-relevance proxy is exploratory and does not replace perceptual validation. For affricates the proxy substitutes the detected frication body of one context for another and compares the junction with the same token's late frication; the release burst is not part of the substituted segment.

## Status and limitations

v0.0.1 is the first working core analysis milestone. Existing Mandarin voicebanks can be inspected, analyzed, converted to a neutral speaker profile, and compiled into a synthesis-unit inventory.

Current decision rules and thresholds are provisional. They were developed during early validation and should not be interpreted as universal phonetic thresholds or as proof of perceptual equivalence. Results should be treated as research-oriented evidence for recording-scheme design rather than as definitive phonological classification.

Not implemented yet:

- final UTAU/OpenUtau alias compilation
- adaptive `oto.ini` suggestions
- merging supplementary diagnostic items into the compiled reclist
- evidence-based removal of VC connections
- broad cross-speaker validation
- formal perceptual validation
- general language-module support beyond the current Mandarin path

## Data and privacy

PhonoWeave analyzes local voicebank files supplied by the user. This repository does not include third-party voicebanks, test recordings, or bundled speech datasets. Users are responsible for ensuring they have permission to analyze and redistribute any external voicebank or recording material they use with the project.

Generated profiles may contain the source voicebank path. Review generated artifacts before publishing them if the local path contains identifying information.

## Project structure

```text
phonoweave/       core analysis and compilation code
languages/        language-specific inventories and rules
schemas/          shared data formats
docs/             design notes and architecture
tests/            tests
```

## Development

Run the test suite with:

```bash
pytest -q
```

GitHub Actions runs the test suite on supported Python versions for pushes to `main` and for pull requests.

## License

PhonoWeave is released under the MIT License. See `LICENSE`.
