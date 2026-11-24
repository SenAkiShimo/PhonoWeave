from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bank_audit import audit_bank, format_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phonoweave audit-bank")
    parser.add_argument("voicebank", type=Path)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args(argv)

    audit = audit_bank(args.voicebank)
    print(format_audit(audit))
    if args.json_path:
        args.json_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
