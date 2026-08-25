from __future__ import annotations

import argparse
from pathlib import Path

from .source_audit import audit_sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phonoweave.source_identity_cli")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--base", action="append", dest="bases")
    args = parser.parse_args(argv)

    bases = tuple(args.bases or ("b", "p", "d", "t", "g", "k"))
    result = audit_sources(args.voicebank, bases)

    print(f"Voicebank: {result.voicebank}")
    print(f"Requested bases: {','.join(bases)}")
    print(f"Shared acoustic segments: {len(result.shared_segments)}")
    print()

    if not result.shared_segments:
        print("none")
        return 0

    for segment in result.shared_segments:
        print(
            f"{segment.status}: observations={segment.observations}, "
            f"wav={segment.wav_path}, "
            f"segment={segment.start_ms:.3f}..{segment.end_ms:.3f} ms"
        )
        for label in segment.labels:
            print(
                f"  {label.base_unit}/{label.final} role={label.role} "
                f"oto_set={label.oto_set} alias={label.alias!r}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
