from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError
from .rhotic_splice import (
    RhoticSpliceSample,
    _collect,
    _median_penalty,
    _other,
    rhotic_boundary_penalty,
)


_CONTEXTS = ("plain", "front", "rounded")


@dataclass(frozen=True)
class DirectionalScore:
    subbank: str
    target_context: str
    donor_context: str
    targets: int
    natural: float
    same_control: float
    donor_penalty: float
    delta_vs_same: float
    excess_over_natural: float


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _score_direction(
    subbank: str,
    target_context: str,
    donor_context: str,
    targets: list[RhoticSpliceSample],
    donors: list[RhoticSpliceSample],
) -> DirectionalScore | None:
    natural_values: list[float] = []
    same_values: list[float] = []
    donor_values: list[float] = []

    for target in targets:
        same_donors = _other(targets, target)
        if not same_donors:
            continue
        directional_donors = _other(donors, target) if donor_context == target_context else donors
        if not directional_donors:
            continue
        try:
            natural = rhotic_boundary_penalty(target.core, target.late)
            same = _median_penalty(same_donors, target)
            donor = same if donor_context == target_context else _median_penalty(directional_donors, target)
        except (AudioReadError, ValueError):
            continue
        natural_values.append(natural)
        same_values.append(same)
        donor_values.append(donor)

    if not donor_values:
        return None

    natural = _mean(natural_values)
    same = _mean(same_values)
    donor = _mean(donor_values)
    return DirectionalScore(
        subbank=subbank,
        target_context=target_context,
        donor_context=donor_context,
        targets=len(donor_values),
        natural=natural,
        same_control=same,
        donor_penalty=donor,
        delta_vs_same=donor - same,
        excess_over_natural=donor - natural,
    )


def analyze_directional_matrix(root: Path) -> list[DirectionalScore]:
    root = root.expanduser().resolve()
    grouped = _collect(root)
    rows: list[DirectionalScore] = []

    for subbank in sorted(grouped):
        for target_context in _CONTEXTS:
            targets = grouped[subbank].get(target_context, [])
            if len(targets) < 2:
                continue
            for donor_context in _CONTEXTS:
                donors = grouped[subbank].get(donor_context, [])
                if len(donors) < 2:
                    continue
                row = _score_direction(
                    subbank,
                    target_context,
                    donor_context,
                    targets,
                    donors,
                )
                if row is not None:
                    rows.append(row)
    return rows


def _aggregate(rows: list[DirectionalScore]) -> dict[tuple[str, str], tuple[float, float, float]]:
    grouped: dict[tuple[str, str], list[DirectionalScore]] = defaultdict(list)
    for row in rows:
        grouped[(row.target_context, row.donor_context)].append(row)

    output: dict[tuple[str, str], tuple[float, float, float]] = {}
    for key, values in grouped.items():
        output[key] = (
            _mean([value.donor_penalty for value in values]),
            _mean([value.delta_vs_same for value in values]),
            _mean([value.excess_over_natural for value in values]),
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.rhotic_substitution")
    parser.add_argument("voicebank", type=Path)
    args = parser.parse_args()

    rows = analyze_directional_matrix(args.voicebank)
    overall = _aggregate(rows)

    print("Rhotic directional join matrix")
    print("Rows are target contexts; columns are donor contexts.")
    print("Cell = delta vs same-context control. Lower is better.")
    print()

    header = "target \\ donor".ljust(18) + "".join(context.rjust(12) for context in _CONTEXTS)
    print(header)
    for target in _CONTEXTS:
        line = target.ljust(18)
        for donor in _CONTEXTS:
            value = overall.get((target, donor))
            line += (f"{value[1]:+0.4f}" if value is not None else "n/a").rjust(12)
        print(line)

    print()
    print("Per subbank:")
    for subbank in sorted({row.subbank for row in rows}):
        print(f"  {subbank}")
        bank_rows = [row for row in rows if row.subbank == subbank]
        bank = _aggregate(bank_rows)
        for target in _CONTEXTS:
            values = []
            for donor in _CONTEXTS:
                value = bank.get((target, donor))
                values.append(f"{donor}={value[1]:+0.4f}" if value is not None else f"{donor}=n/a")
            print(f"    {target}: " + ", ".join(values))

    print()
    print("Directional canonical-donor summary:")
    for donor in _CONTEXTS:
        deltas = [
            overall[(target, donor)][1]
            for target in _CONTEXTS
            if target != donor and (target, donor) in overall
        ]
        if deltas:
            print(f"  {donor}: mean cross-target join delta={_mean(deltas):+0.4f}")

    print()
    print("This matrix measures join compatibility only, not realization correctness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
