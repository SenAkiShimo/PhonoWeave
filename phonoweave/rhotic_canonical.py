from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioReadError
from .rhotic_splice import _collect, _median_penalty, _other


@dataclass(frozen=True)
class CanonicalTarget:
    subbank: str
    target_context: str
    donor_context: str
    alias: str
    same_control: float
    donor_penalty: float
    delta: float


@dataclass(frozen=True)
class CanonicalSubbankResult:
    subbank: str
    targets: int
    mean_delta: float
    permutation_p: float


@dataclass(frozen=True)
class CanonicalDirectionResult:
    target_context: str
    donor_context: str
    targets: int
    mean_same_control: float
    mean_donor_penalty: float
    mean_delta: float
    permutation_p: float
    subbanks: list[CanonicalSubbankResult]


@dataclass(frozen=True)
class RhoticCanonicalAnalysis:
    plain_to_rounded: CanonicalDirectionResult | None
    rounded_to_plain: CanonicalDirectionResult | None


def _paired_p(values: list[float], seed: int) -> float:
    deltas = np.asarray(values, dtype=np.float64)
    if len(deltas) == 0:
        return 1.0
    observed = float(np.mean(deltas))
    if observed <= 0:
        return 1.0

    rng = np.random.default_rng(seed)
    exceed = 0
    permutations = 10000
    for _ in range(permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(deltas))
        if float(np.mean(deltas * signs)) >= observed:
            exceed += 1
    return float((exceed + 1) / (permutations + 1))


def _direction(
    grouped,
    target_context: str,
    donor_context: str,
) -> list[CanonicalTarget]:
    rows: list[CanonicalTarget] = []
    for subbank in sorted(grouped):
        targets = grouped[subbank].get(target_context, [])
        donors = grouped[subbank].get(donor_context, [])
        if len(targets) < 2 or len(donors) < 2:
            continue
        for target in targets:
            same_donors = _other(targets, target)
            if not same_donors:
                continue
            try:
                same = _median_penalty(same_donors, target)
                donor = _median_penalty(donors, target)
            except (AudioReadError, ValueError):
                continue
            rows.append(
                CanonicalTarget(
                    subbank=subbank,
                    target_context=target_context,
                    donor_context=donor_context,
                    alias=target.alias,
                    same_control=same,
                    donor_penalty=donor,
                    delta=donor - same,
                )
            )
    return rows


def _mean(rows: list[CanonicalTarget], field: str) -> float:
    return float(np.mean([getattr(row, field) for row in rows]))


def _summarize(
    rows: list[CanonicalTarget],
    target_context: str,
    donor_context: str,
    seed: int,
) -> CanonicalDirectionResult | None:
    if not rows:
        return None

    subbanks: list[CanonicalSubbankResult] = []
    for subbank in sorted({row.subbank for row in rows}):
        bank = [row for row in rows if row.subbank == subbank]
        subbanks.append(
            CanonicalSubbankResult(
                subbank=subbank,
                targets=len(bank),
                mean_delta=_mean(bank, "delta"),
                permutation_p=_paired_p(
                    [row.delta for row in bank],
                    seed + len(subbank),
                ),
            )
        )

    return CanonicalDirectionResult(
        target_context=target_context,
        donor_context=donor_context,
        targets=len(rows),
        mean_same_control=_mean(rows, "same_control"),
        mean_donor_penalty=_mean(rows, "donor_penalty"),
        mean_delta=_mean(rows, "delta"),
        permutation_p=_paired_p([row.delta for row in rows], seed),
        subbanks=subbanks,
    )


def analyze_rhotic_canonical(root: Path) -> RhoticCanonicalAnalysis:
    grouped = _collect(root.expanduser().resolve())
    plain_to_rounded_rows = _direction(grouped, "rounded", "plain")
    rounded_to_plain_rows = _direction(grouped, "plain", "rounded")
    return RhoticCanonicalAnalysis(
        plain_to_rounded=_summarize(
            plain_to_rounded_rows,
            "rounded",
            "plain",
            11213,
        ),
        rounded_to_plain=_summarize(
            rounded_to_plain_rows,
            "plain",
            "rounded",
            11239,
        ),
    )


def _print_direction(title: str, result: CanonicalDirectionResult) -> None:
    print(title)
    print(f"  targets: {result.targets}")
    print(f"  same-context control: {result.mean_same_control:.4f}")
    print(f"  canonical donor penalty: {result.mean_donor_penalty:.4f}")
    print(f"  delta vs same: {result.mean_delta:+.4f}")
    print(f"  permutation p: {result.permutation_p:.4f}")
    for item in result.subbanks:
        print(
            f"  {item.subbank}: delta={item.mean_delta:+.4f}, "
            f"p={item.permutation_p:.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.rhotic_canonical")
    parser.add_argument("voicebank", type=Path)
    args = parser.parse_args()

    result = analyze_rhotic_canonical(args.voicebank)

    print("Rhotic canonical-donor test")
    print("Tests one-way reuse for the plain/rounded merge candidate.")
    print()

    if result.plain_to_rounded is not None:
        _print_direction(
            "Canonical plain -> rounded targets:",
            result.plain_to_rounded,
        )
        print()
    if result.rounded_to_plain is not None:
        _print_direction(
            "Canonical rounded -> plain targets:",
            result.rounded_to_plain,
        )

    print()
    print("Interpretation: a small delta and weak evidence of harm support that donor direction.")
    print("This is not an equivalence test and does not replace realization evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
