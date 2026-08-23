from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .mandarin import collect_observations, context_for
from .oto import OtoEntry, load_voicebank
from .prefixmap import affix_pairs, load_prefix_maps


def _subbank_name(root: Path, entry: OtoEntry) -> str:
    directory = entry.oto_path.parent
    return "." if directory == root else str(directory.relative_to(root))


def _segment_key(entry: OtoEntry) -> tuple[Path, float, float]:
    return (
        entry.wav_path,
        round(entry.offset, 3),
        round(entry.offset + entry.preutterance, 3),
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.rhotic_samples")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument(
        "--context",
        choices=("plain", "front", "rounded", "all"),
        default="front",
    )
    args = parser.parse_args()

    root = args.voicebank.expanduser().resolve()
    entries, _ = load_voicebank(root)
    valid_entries = [entry for entry in entries if entry.wav_path.exists()]
    affixes = affix_pairs(load_prefix_maps(root))
    observations = collect_observations(valid_entries, affixes)

    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        if observation.base_unit != "r":
            continue
        context = context_for("r", observation.final)
        if context is None:
            continue
        grouped[_subbank_name(root, observation.entry)][context].append(observation)

    print("Rhotic source observations:")
    for subbank in sorted(grouped):
        print(f"  {subbank}")
        for context in ("plain", "front", "rounded"):
            rows = grouped[subbank].get(context, [])
            unique_wavs = {row.entry.wav_path for row in rows}
            unique_segments = {_segment_key(row.entry) for row in rows}
            finals = sorted({row.final for row in rows})
            print(
                f"    {context}: observations={len(rows)}, "
                f"unique_wavs={len(unique_wavs)}, "
                f"unique_segments={len(unique_segments)}, "
                f"finals={','.join(finals) or '-'}"
            )
    print()

    contexts = ("plain", "front", "rounded") if args.context == "all" else (args.context,)
    for subbank in sorted(grouped):
        for context in contexts:
            rows = grouped[subbank].get(context, [])
            if not rows:
                continue
            print(f"[{subbank} / {context}]")
            rows = sorted(
                rows,
                key=lambda row: (
                    row.entry.wav_path.name,
                    row.entry.offset,
                    row.entry.preutterance,
                    row.entry.alias,
                ),
            )
            segment_groups: dict[tuple[Path, float, float], list] = defaultdict(list)
            for row in rows:
                segment_groups[_segment_key(row.entry)].append(row)

            for row in rows:
                entry = row.entry
                start = entry.offset
                end = entry.offset + entry.preutterance
                duplicate_count = len(segment_groups[_segment_key(entry)])
                duplicate = f" shared_segment={duplicate_count}" if duplicate_count > 1 else ""
                print(
                    f"  final={row.final:<5} alias={entry.alias!r} "
                    f"wav={entry.wav_path.name!r} "
                    f"segment={start:.3f}-{end:.3f}ms "
                    f"oto_line={entry.line_number}{duplicate}"
                )
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
