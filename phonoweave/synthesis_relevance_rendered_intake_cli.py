"""CLI for frozen render tasks and independently verified blind A/B packaging."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets

from .synthesis_relevance_rendered_intake import (
    create_render_plan, stage_blind_session, write_render_plan,
)


def _json(path: Path) -> dict:
    with path.expanduser().open(encoding="utf-8") as stream:
        return json.load(stream)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="phonoweave-rendered-intake",
        description="Prepare S01 donor routes or blind externally rendered S02 WAVs",
    )
    commands = p.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("manifest", type=Path)
    plan.add_argument("--proxy", type=Path, required=True)
    plan.add_argument("--root", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    blind = commands.add_parser("blind")
    blind.add_argument("--plan", type=Path, required=True)
    blind.add_argument("--receipt", type=Path, required=True)
    blind.add_argument("--renders", type=Path, required=True)
    blind.add_argument("--corpus", type=Path, required=True)
    blind.add_argument("--output", type=Path, required=True)
    blind.add_argument("--private-key", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.action == "plan":
            manifest = args.manifest.expanduser().resolve(strict=True)
            corpus = args.root or manifest.parent
            result = create_render_plan(_json(manifest), _json(args.proxy), corpus)
            write_render_plan(result, args.output, corpus=corpus)
            print(f"Frozen external render tasks: {result['task_count']}")
            print("No rendering performed; source session S02 is reference-only.")
        else:
            seed = secrets.token_hex(32)
            plan_data = _json(args.plan)
            receipt = _json(args.receipt)
            participant, _ = stage_blind_session(
                plan_data, receipt, args.renders, args.output,
                args.private_key, corpus=args.corpus, seed_hex=seed,
            )
            print(f"Blind trials: {len(participant['trials'])}")
            print("Participant media: ", args.output)
            print("Private key: ", args.private_key)
            print("Keep the private key separate from participants.")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
