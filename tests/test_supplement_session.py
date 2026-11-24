from __future__ import annotations

import pytest

from phonoweave.supplement_session import (
    create_session,
    load_session,
    parse_layers,
    parse_lines,
    save_take,
    take_path,
)

WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 40


def test_lines_and_layers_are_parsed() -> None:
    assert parse_lines("ba_nuo_nu\n\n  zu zuo \n") == [("ba", "nuo", "nu"), ("zu", "zuo")]
    assert parse_layers("G3, D4  G4 G4") == ["G3", "D4", "G4"]
    with pytest.raises(ValueError):
        parse_layers("../x")


def test_a_take_is_saved_and_found_again(tmp_path) -> None:
    directory = create_session(tmp_path, [("ba", "nuo"), ("zu", "zuo")], ["G4", "D5"], 90)

    save_take(directory, "G4", 1, WAV, 48000, [100.0, 800.0])

    loaded = load_session(directory)
    assert loaded["completed"] == {"G4": [1], "D5": []}
    assert take_path(directory, "G4", 1).read_bytes() == WAV
    with pytest.raises(ValueError):
        take_path(directory, "D5", 1)


def test_bad_requests_are_refused(tmp_path) -> None:
    directory = create_session(tmp_path, [("ba", "nuo")], ["G4"], 90)

    with pytest.raises(ValueError):
        save_take(directory, "G4", 0, b"not a wav", 48000, [0.0, 1.0])
    with pytest.raises(ValueError):
        save_take(directory, "C9", 0, WAV, 48000, [0.0, 1.0])
    with pytest.raises(ValueError):
        save_take(directory, "G4", 3, WAV, 48000, [0.0, 1.0])
    with pytest.raises(ValueError):
        save_take(directory, "G4", 0, WAV, 48000, [0.0])
    with pytest.raises(ValueError):
        create_session(tmp_path, [("ba",)], ["G4"], 5)
