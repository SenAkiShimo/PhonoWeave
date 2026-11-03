"""Run the S01-only executable reference splice renderer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .synthesis_relevance_reference_renderer import render_reference_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render real WAVs with the experimental S01-only splice baseline"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.expanduser().read_text(encoding="utf-8"))
        manifest = json.loads(args.manifest.expanduser().read_text(encoding="utf-8"))
        result = render_reference_plan(plan, manifest, args.corpus, args.output)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print("Generated reference-splice WAVs:", result["rendered_wavs"])
    print("Receipt:", args.output / "renderer_receipt.json")
    print("Reference audio baseline only; NOT a singing resampler.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
