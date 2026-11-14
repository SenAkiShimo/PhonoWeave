from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .reclist_cli import compile_reclist, print_summary
from .starter import starter_inventory, starter_syllables
from .synthesis_inventory import synthesis_inventory_yaml


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave starter-reclist")
    parser.add_argument("-o", "--output", type=Path, default=Path("starter_reclist.txt"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-syllables", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-vc", choices=("shared", "split"), default="shared")
    args = parser.parse_args(argv)

    syllables = starter_syllables()
    inventory = starter_inventory()
    try:
        compiled = compile_reclist(
            syllables,
            inventory,
            args.output,
            args.report,
            args.max_syllables,
            args.restarts,
            args.seed,
            "collapse",
            args.stop_vc,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    scheme = compiled.output.with_name(compiled.output.stem + ".units.yaml")
    scheme.write_text(synthesis_inventory_yaml(inventory), encoding="utf-8")
    print_summary(compiled, len(syllables), "collapse")
    print(f"Unit scheme: {scheme}")
    print("Prior scheme from general Mandarin rules; not tested on any speaker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
