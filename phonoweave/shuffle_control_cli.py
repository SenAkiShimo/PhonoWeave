from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .inventory import analyze_voicebank_inventory
from .shuffle_control import build_shuffled_copy, split_calls, summarize


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave shuffle-control")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--single-layer-evidence", action="store_true")
    args = parser.parse_args(argv)

    real = split_calls(analyze_voicebank_inventory(args.voicebank, args.single_layer_evidence).decisions)
    runs = []
    for run in range(args.runs):
        with tempfile.TemporaryDirectory() as scratch:
            copy = Path(scratch) / "bank"
            changed = build_shuffled_copy(args.voicebank, copy, args.seed + run)
            runs.append(split_calls(analyze_voicebank_inventory(copy, args.single_layer_evidence).decisions))
        print(f"run {run + 1}/{args.runs}: {changed} aliases relabeled", flush=True)

    text = "\n".join(summarize(real, runs)) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    return 0
