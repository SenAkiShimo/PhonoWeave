# Reclist compiler

`phonoweave build-reclist` compiles a speaker's synthesis inventory into a continuous recording list. The coverage target is derived from the inventory, not from a fixed onset-by-final grid.

## Inputs

- Syllables recognized in the existing voicebank (`oto.ini` aliases, after prefix-map normalization).
- The synthesis inventory, either produced on the fly or loaded from `synthesis_inventory.yaml`.

## Model

Each recognized syllable is assigned an onset unit by looking up its onset and final in the inventory. Units are matched through the context labels stored in the profile (`plain`, `rounded`, `front`, `front_unrounded`, and role-scoped `internal:` / `initial:` families). Onsets absent from the inventory use one unit per onset. Zero-onset syllables share the unit `0`.

A line is a sequence of syllables. Position determines the role of each onset: the first syllable is `initial`, the others are `internal`. Coverage requirements are:

| Requirement | Satisfied when |
| --- | --- |
| CV | the syllable occurs with the onset unit assigned for that role |
| VC | a syllable ending in final `f` is followed by a syllable whose onset unit is `u` |
| start | a line begins with a syllable of onset unit `u` |
| end | a line ends with final `f` |

The VC set is every observed final crossed with every internal onset unit. A split decision therefore adds connections for the new unit, and a merge decision keeps them shared. Onset/final pairs that the inventory cannot place are listed under `unmapped_contexts` in the report and fall back to a per-onset unit.

## Unresolved decisions

Groups whose decision is `unresolved` carry no evidence for a split. By default (`--unresolved collapse`) they are replaced by one unit per onset, so they cost the same as in a conventional list. With `--unresolved split` every unresolved group stays a separate unit, which keeps the recordings usable if a later analysis supports a split but multiplies the VC requirements. The report lists the collapsed onsets and gives the line count of the other mode under `alternative`.

## Optimization

Coverage is built from a balanced multigraph. Onset units and finals are nodes; every recognized syllable is a unit-to-final edge that must be used at least once, and every required VC is a final-to-unit edge. A minimum-cost flow adds the fewest extra syllable and connection uses that make the graph balanced, an Euler circuit through it visits every requirement, and the circuit is cut into lines of at most `--max-syllables` syllables that overlap by one syllable so that no connection is lost at a cut. Cut points are chosen so that utterance-start and utterance-end requirements are also met; anything still missing is added as short lines, and redundant lines are removed at the end.

The report gives a lower bound computed from the requirement counts and the line limit. The search does not prove optimality, but on dense Mandarin-sized inputs it stays within a few lines of the bound. Every produced list is checked against the requirement sets before it is written.

## Limitations

- Final-side classes are not merged. Every observed final is its own VC source.
- No VC connection is removed on acoustic grounds yet. Doing so needs a transition analysis that the profile does not contain; `build_reclist` accepts a `waived_vc` set for when that evidence exists.
- Nasal and lateral split decisions are currently written to the profile as `plain` / `rounded` groups, which the compiler matches through the rounding context of the final.
- Coverage of recognized symbols says nothing about untested acoustic variants or perceptual quality.

## Transition consistency (exploratory)

`phonoweave transition-report` compares consonant-to-vowel trajectories across context families (i_series, u_series, other) for one onset.

- Frames of 25 ms every 5 ms, from 50 ms before to 100 ms after the preutterance boundary.
- Each frame is reduced to a cepstrally smoothed log envelope on 24 log-spaced bands (150 Hz to 4.5 kHz), with the frame mean removed, so harmonic spacing and level do not enter.
- The consonant state is the mean envelope 50 to 25 ms before the boundary; the vowel state is the mean 60 to 100 ms after it. Samples with less than 3 dB RMS contrast between the two are skipped.
- Progress along the line from consonant state to vowel state is computed per frame. The component of the path off that line is read out where progress first reaches 0.25, 0.5 and 0.75.
- Distances are compared within each pitch layer only. The statistic is the mean between-family distance minus the mean within-family distance; labels are permuted within layers, and Holm correction is applied across family pairs and progress levels.

Limits: the residual still depends on the vowel through the non-linear shape of the path, so the same-final and different-final within-family distances are printed as a reference; samples from one session are not independent; the result is not a perceptual measurement and does not by itself justify merging contexts.
