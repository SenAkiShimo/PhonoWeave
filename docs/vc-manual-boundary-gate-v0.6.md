# Manual VC transition boundary gate (v0.6)

## Why this exists
The existing 16ms OTO window is not a verified consonant boundary: in the observed LynneQingD4 `sh` examples frication energy increases around 20–40ms after the old anchor. It would be scientifically invalid to reuse all 830 provisional feature-bearing OTO windows as reliable phonetic segments.

This branch introduces two steps, **without adding another split/merge heuristic**.

1. `phonoweave vc-review inspect --pack /path/to/v0.5_pack --output /outside/new_proposals.json` verifies WAV hashes against the original v0.5 manifest, and suggests *unverified* sustained high-frequency noise-onset brackets for /sh/. /m/ abstains because nasal transitions cannot be discovered using frication thresholds. The diagnostic thresholds are exploratory, not gold accuracy.
2. Mark the real start and end of each VC transition by listening and reviewing waveform/spectrogram. Then call `phonoweave vc-review validate --pack /path/to/v0.5_pack --annotations completed.csv --output /outside/validated_review.json`. The CLI verifies identities, six rows, valid intervals and explicit reviewer attribution, but **never authorizes split/share**. If not all six clear targets are reviewed, report remains incomplete/uncertain.

### Experimental stop condition
No automatic split/share training from the old OTO-midpoint spectrum; no efficacy claim using same-WAV windows as independent experimental repetitions. Independent recorded takes and held-out synthesis evaluation are still essential. Preserve all original WAV, OTO, aliases, phonetic profiles and source hashes.

Generated material from the author's local real voicebank is never committed to GitHub. This PR contains only tools, tests and documentation.
