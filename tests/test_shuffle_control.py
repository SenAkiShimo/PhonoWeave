from __future__ import annotations

from pathlib import Path

from phonoweave.oto import load_voicebank
from phonoweave.shuffle_control import build_shuffled_copy, summarize


def _bank(root: Path) -> Path:
    bank = root / "bank"
    (bank / "D4").mkdir(parents=True)
    (bank / "prefix.map").write_text("D4\tD4\t\n", encoding="utf-8")
    names = ["sha", "shi", "shu", "shuo", "she", "shan", "ba", "bi"]
    lines = []
    for name in names:
        (bank / "D4" / f"{name}.wav").write_bytes(b"")
        lines.append(f"{name}.wav=D4{name},0,10,0,5,2\n")
    (bank / "D4" / "a_sh.wav").write_bytes(b"")
    lines.append("a_sh.wav=D4a sh,0,10,0,5,2\n")
    (bank / "D4" / "oto.ini").write_text("".join(lines), encoding="utf-8")
    return bank


def test_shuffle_keeps_audio_onset_and_vc(tmp_path) -> None:
    bank = _bank(tmp_path)
    changed = build_shuffled_copy(bank, tmp_path / "copy", seed=3)

    original, _ = load_voicebank(bank)
    shuffled, _ = load_voicebank(tmp_path / "copy")
    before = {e.wav_path.name: e.alias for e in original}
    after = {e.wav_path.name: e.alias for e in shuffled}

    assert changed > 0
    assert sorted(before.values()) == sorted(after.values())
    assert after["a_sh.wav"] == "D4a sh"
    assert all(a.startswith("D4sh") for n, a in after.items() if n.startswith("sh"))
    assert all(a.startswith("D4b") for n, a in after.items() if n.startswith("b"))
    assert any(before[n] != after[n] for n in before)


def test_summary_reports_false_split_rate() -> None:
    real = {"sh": "split_recommended", "s": "unresolved"}
    runs = [{"sh": "unresolved", "s": "split_recommended"}, {"sh": "unresolved", "s": "unresolved"}]
    text = "\n".join(summarize(real, runs))

    assert "sh      split_recommended   0/2" in text
    assert "1/4 (25%)" in text
