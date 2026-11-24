from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

_LAYER = re.compile(r"^[A-Za-z0-9#_-]{1,24}$")
_UNSAFE = re.compile(r"[^a-z0-9_]+")
MAX_WAV_BYTES = 64 * 1024 * 1024


def parse_lines(text: str) -> list[tuple[str, ...]]:
    lines = []
    for raw in text.splitlines():
        tokens = tuple(token for token in re.split(r"[_\s]+", raw.strip().lower()) if token)
        if tokens:
            lines.append(tokens)
    return lines


def parse_layers(text: str) -> list[str]:
    layers: list[str] = []
    for layer in re.split(r"[\s,]+", text.strip()):
        if not layer:
            continue
        if not _LAYER.match(layer):
            raise ValueError(f"invalid pitch layer name: {layer}")
        if layer not in layers:
            layers.append(layer)
    return layers


def _stem(index: int, tokens: list[str]) -> str:
    return f"{index + 1:02d}_{_UNSAFE.sub('', '_'.join(tokens))[:80]}"


def create_session(
    root: Path,
    lines: list[tuple[str, ...]],
    layers: list[str],
    bpm: int,
) -> Path:
    if not lines:
        raise ValueError("there are no lines to record")
    if not layers:
        raise ValueError("choose at least one pitch layer")
    if not 30 <= bpm <= 240:
        raise ValueError("tempo must be between 30 and 240 BPM")
    directory = root.expanduser().resolve() / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    directory.mkdir(parents=True)
    for layer in layers:
        (directory / layer).mkdir()
    (directory / "session.json").write_text(
        json.dumps(
            {"lines": [list(line) for line in lines], "layers": layers, "bpm": bpm},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return directory


def load_session(directory: Path) -> dict:
    directory = directory.expanduser().resolve()
    path = directory / "session.json"
    if not path.is_file():
        raise ValueError("this folder is not a recording session")
    session = json.loads(path.read_text(encoding="utf-8"))
    completed = {
        layer: sorted(
            index
            for index, line in enumerate(session["lines"])
            if (directory / layer / f"{_stem(index, line)}.wav").is_file()
        )
        for layer in session["layers"]
    }
    return {**session, "directory": str(directory), "completed": completed}


def _locate(directory: Path, layer: str, index: int) -> tuple[dict, Path]:
    session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    if layer not in session["layers"]:
        raise ValueError("unknown pitch layer")
    if not 0 <= index < len(session["lines"]):
        raise ValueError("unknown line")
    stem = _stem(index, session["lines"][index])
    return session, directory / layer / f"{stem}.wav"


def take_path(directory: Path, layer: str, index: int) -> Path:
    path = _locate(directory.expanduser().resolve(), layer, index)[1]
    if not path.is_file():
        raise ValueError("this line has not been recorded")
    return path


def save_take(
    directory: Path,
    layer: str,
    index: int,
    wav_bytes: bytes,
    sample_rate: int,
    beats_ms: list[float],
) -> Path:
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("recording is not a WAV file")
    if len(wav_bytes) > MAX_WAV_BYTES:
        raise ValueError("recording is too large")
    directory = directory.expanduser().resolve()
    session, path = _locate(directory, layer, index)
    tokens = session["lines"][index]
    if len(beats_ms) != len(tokens):
        raise ValueError("beat times do not match the line")
    path.write_bytes(wav_bytes)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "tokens": tokens,
                "layer": layer,
                "sample_rate": sample_rate,
                "bpm": session["bpm"],
                "beats_ms": beats_ms,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
