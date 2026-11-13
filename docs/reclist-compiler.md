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

## Optimization

Lines are built by randomized greedy search with restarts and then pruned of redundant lines. The report gives a lower bound computed from the requirement counts and the line limit (`--max-syllables`, default 10). The bound is not tight, and the search does not prove optimality. Every produced list is checked against the requirement sets before it is written.

## Limitations

- Final-side classes are not merged. Every observed final is its own VC source.
- No VC connection is removed on acoustic grounds yet. Doing so needs a transition analysis that the profile does not contain; `build_reclist` accepts a `waived_vc` set for when that evidence exists.
- Nasal and lateral split decisions are currently written to the profile as `plain` / `rounded` groups, which the compiler matches through the rounding context of the final.
- Coverage of recognized symbols says nothing about untested acoustic variants or perceptual quality.
