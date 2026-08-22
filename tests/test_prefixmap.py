from pathlib import Path

from phonoweave.prefixmap import load_prefix_maps


def test_load_prefix_map(tmp_path: Path) -> None:
    (tmp_path / "prefix.map").write_text(
        "C3\t\t_C3\nC#3\t\t_C3\nD3\t\t_D3\n",
        encoding="utf-8",
    )

    rules = load_prefix_maps(tmp_path)

    assert len(rules) == 2
    by_suffix = {rule.suffix: rule for rule in rules}
    assert by_suffix["_C3"].tone_ranges == ("C3-C#3",)
    assert by_suffix["_D3"].tone_ranges == ("D3",)
