from __future__ import annotations

import argparse
from pathlib import Path

from .bank_audit import audit_bank
from .bank_concordance import compare_with_bank, format_concordance
from .inventory import analyze_voicebank_inventory
from .reclist import observed_syllables
from .synthesis_inventory import build_synthesis_inventory, load_synthesis_inventory
from .profile import build_speaker_profile
from .tolerance import apply_tolerance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave compare-bank")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--tolerance", choices=("strict", "balanced", "loose"), default="strict")
    args = parser.parse_args(argv)

    if args.inventory is not None:
        inventory = load_synthesis_inventory(args.inventory)
    else:
        analysis = analyze_voicebank_inventory(args.voicebank)
        profile = build_speaker_profile(args.voicebank, analysis)
        inventory = build_synthesis_inventory(args.voicebank, profile)

    inventory, _ = apply_tolerance(inventory, args.tolerance)
    rows = compare_with_bank(
        audit_bank(args.voicebank),
        inventory,
        observed_syllables(args.voicebank),
        args.tolerance,
    )
    print(format_concordance(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
