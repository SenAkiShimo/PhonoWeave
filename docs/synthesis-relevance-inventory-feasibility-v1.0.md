# Synthesis relevance v1.0 — physical WAV inventory and counterfactual feasibility

## Objective

Determine why v0.9's prespecified exact-final donor design produced zero valid targets across eight existing voicebanks. This phase audits **replication feasibility**, not acoustic effect, rendered singing quality, or split/merge necessity.

## Inventory and replication unit

Extract the same eligible OTO/WAV units as v0.8/v0.9, preserving all original audio and OTO parameters. Group by **bank / OTO subbank / exact phonetic final** and report the number of distinct physical source WAVs, OTO event count, and the duration range. Two OTO aliases from the same WAV do not count as independent repetitions. WAVs from different pitch subbanks are counted in those subbanks separately, not implicitly interchangeable.

The four frozen final pairs remain `a/ua`, `ai/uai`, `an/uan`, `ang/uang`. A held-out target requires a different-WAV same-final donor, a different-WAV opposite-final donor from its frozen pair, a 10-ms duration caliper in both arms and no more than 5-ms difference in their time-matching errors.

## Exclusive reason diagnosis

Every eligible unit receives exactly one of: outside final pairs; no independent same-final donor; no independent opposite-final donor; same-final duration-caliper failure; opposite-final duration-caliper failure; arm-duration imbalance; valid strict target. The report includes closest independent-donor duration deviations when they exist and verifies the final valid count against frozen v0.9 `matched_trials`. These blocker categories represent the first failed sequential gate; later gates may also be unsatisfied.

## Recording additions: explicitly bounded planning quantity

For **two distinct physical WAV instances per final** as a structural baseline, record the deficit per final as `max(0,2-N_WAV)`. The sum of deficits across both sides of a final pair is a lower bound for added final recording *instances*. If a new physical WAV contains both target finals, the pair-level lower bound in added physical WAVs is the **maximum** of the two deficits, not their sum. These figures are not a guaranteed 10-ms matched design, not proven minimum numbers of new recording sessions, and cannot be added across multiple pairs to claim a corpus-wide physical-WAV optimum.

Separate repeated-session/speaker requirements, registration, verified frication boundaries, and duration-aligned samples would be needed for reliable experimental counterfactual inference. Do not produce an inventory split decision from these counts.

## Run

```bash
cd ~/Downloads/PhonoWeave
git fetch origin
git switch --track origin/research/synthesis-relevance-inventory-feasibility-v1.0
source .venv/bin/activate
python -m pytest -q
python -m phonoweave.synthesis_relevance_inventory_feasibility_cli \
  ~/.config/phonoweave/voicebanks.json \
  --base sh \
  --output ~/Downloads/phonoweave_sh_inventory_feasibility_v1.0.json
```

The JSON retains exhaustive per-subbank inventories and per-final-pair feasibility. No sound, modified WAV, edited OTO, or listening assessment is produced.
