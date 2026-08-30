from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .calibration_manual_labels import resolve_dev_selection


def _write_manifest(session_dir: Path, output: Path) -> None:
    rows = [
        "# prompt_index\tbase_unit\tcontext_family\tsyllable\tclass_name\twav\toccurrence\tprev_cue_ms\tcue_ms\tnext_cue_ms"
    ]
    for item in resolve_dev_selection(session_dir):
        wav = session_dir / "recordings" / item.wav
        for occurrence in (1, 2):
            i = occurrence - 1
            rows.append(
                "\t".join(
                    [
                        str(item.prompt_index),
                        item.base_unit,
                        item.context_family,
                        item.syllable,
                        item.class_name,
                        str(wav),
                        str(occurrence),
                        f"{item.prev_cues_ms[i]:.3f}",
                        f"{item.cues_ms[i]:.3f}",
                        f"{item.next_cues_ms[i]:.3f}",
                    ]
                )
            )
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonoweave-manual-java")
    parser.add_argument("session", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    session_dir = args.session.expanduser().resolve()
    java = shutil.which("java")
    if not java:
        raise SystemExit("Java was not found. Install a JDK, then run this command again.")

    analysis_dir = session_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    manifest = analysis_dir / "manual_anchor_manifest_v0.2.tsv"
    labels = analysis_dir / "calibration_manual_anchor_labels_v0.2.tsv"
    _write_manifest(session_dir, manifest)

    repo_root = Path(__file__).resolve().parent.parent
    source = repo_root / "tools" / "ManualAnchorWebV2.java"
    if not source.is_file():
        raise SystemExit(f"Java web labeler source is missing: {source}")

    env = os.environ.copy()
    # ManualAnchorWebV2 starts phonoweave.gui as a child process when the user
    # clicks the return button. Python normally block-buffers stdout when it is
    # piped, which prevents Java from seeing the GUI URL. Make that child
    # interpreter unbuffered so the URL is available immediately.
    env["PYTHONUNBUFFERED"] = "1"

    completed = subprocess.run(
        [
            java,
            "--add-modules",
            "jdk.httpserver",
            str(source),
            str(manifest),
            str(labels),
            sys.executable,
            str(repo_root),
        ],
        check=False,
        env=env,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
