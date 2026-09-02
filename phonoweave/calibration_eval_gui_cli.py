from __future__ import annotations

import argparse
from pathlib import Path

from .calibration_eval_gui import main as gui_main


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-calibration-eval-gui")
    parser.add_argument("session", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    forwarded = [
        "--calibration-session",
        str(args.session.expanduser().resolve()),
        "--port",
        str(args.port),
    ]
    if args.no_browser:
        forwarded.append("--no-browser")
    return gui_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
