# Synthesis Relevance v0.4 — blinded OTO-relative cut sensitivity

## Research question

The first rendered hybrid CV trial had an audible /shw/-like onset in the cross-context condition but all hybrid variants sounded somewhat unnatural. The origin of this difference is ambiguous: it may reflect naturally anticipatory rounding in the fricative, or unintentional retention of following glide/vowel content at the provisional OTO cut. This pilot varies the left **donor cut coordinate**, while preserving the same physical target WAV and two donor WAVs selected in the prior v0.3 pilot.

## Frozen stimulus construction

- Source identities come from v0.3 `experiment.json`. The target and both donors must belong to the same subbank and have **distinct physical source WAVs**.
- A common target post-boundary audio segment of 600 ms is used throughout; the *right* boundary always stays fixed at the original target `offset+preutterance`.
- Two source contexts: matched and cross. Each donor's **left-hand window** ends at OTO `offset+preutterance` shifted by **−40, −20 or 0 ms**. Its duration is fixed to a conservative pre-boundary range for **all** conditions. Each joint uses the same 8 ms equal-power crossfade and single shared gain coefficient.
- Six A/B-alias WAVs are randomly assigned opaque names `PW_T01` through `PW_T06`, with the semantic mapping written into a separate `<singer>_REVEAL.json` file located outside the singer directory. A `PW_REF` alias contains a continuous slice of the target natural source; this is **not** the main treatment/control arm.
- Output includes only the six opaque aliases, `PW_REF`, `oto.ini`, `character.txt`, a non-revealing `experiment.json`, and an external reveal file with original relative WAV/event sources and hashes.

## Interpretation

These are cut-position **offsets relative to an OTO control coordinate**; no cut is called a true acoustically adjudicated end of /sh/. Early cuts can remove part of the frication as well as suspected glide content. Additionally the donor-window onset necessarily shifts earlier with the donor cut: the intervention modifies both which sound is included and its point in the original trajectory. Donor identities are fixed between shifts; cross- and matched-context donor recordings still differ in pitch, phonation, pronunciation and coarticulation. The crossfade is artificial; rendered audio is subject to resampler and note settings. A detectable effect shows that this constructed sample is sensitive to cut position, but cannot by itself prove which underlying phonetic contrast a recording list must maintain.

The experiment **does not** render OpenUtau automatically or interpret listening results, and it never modifies the real voicebank. After staging, inspect the WAVs around the nominal boundary, render each alias at identical OpenUtau settings, randomize listening order without inspecting the reveal key, and document which versions have /w/-like residue, fricative truncation or boundary artifacts. Do not claim equivalence from a failure to hear a difference.

## Usage

```bash
python -m pytest -q
python -m phonoweave.synthesis_relevance_boundary_ablation_cli \
  "/Users/yang/Library/OpenUtau/Singers/LynneQing_CVVChn_Mang" \
  --previous-singer ~/Downloads/PhonoWeave_AB_D4 \
  --output-singer ~/Downloads/PhonoWeave_Boundary_AB_D4
```
