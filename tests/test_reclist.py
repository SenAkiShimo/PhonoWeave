from pathlib import Path

import pytest

from phonoweave.reclist import (
    Syllable,
    UnitMap,
    build_reclist,
    build_requirements,
    lower_bound,
    missing_requirements,
    observed_syllables,
)
from phonoweave.reclist_cli import main as reclist_main
from phonoweave.synthesis_inventory import (
    SynthesisInventory,
    SynthesisUnit,
    load_synthesis_inventory,
    synthesis_inventory_yaml,
)


def _syllable(text: str) -> Syllable:
    for base in ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"):
        if text.startswith(base):
            return Syllable(text=text, onset=base, final=text[len(base):])
    return Syllable(text=text, onset="", final=text)


def _unit(unit_id: str, base: str, *contexts: str, decision: str = "split_recommended") -> SynthesisUnit:
    return SynthesisUnit(
        id=unit_id,
        base_unit=base,
        contexts=contexts,
        source_group=unit_id,
        canonical_context=None,
        decision=decision,
        confidence="high",
        acoustic_evidence="supported",
        synthesis_evidence="supported_under_proxy",
    )


def _inventory(*units: SynthesisUnit) -> SynthesisInventory:
    return SynthesisInventory("test", "mandarin", Path("/tmp/vb"), tuple(units))


SYLLABLES = tuple(
    _syllable(text)
    for text in ("sha", "shu", "shuo", "she", "ba", "bi", "bu", "xi", "xu", "ma", "mi", "a", "ai")
)


def test_split_adds_vc_requirements_and_merge_does_not() -> None:
    base = build_requirements(SYLLABLES, UnitMap(None))
    split = build_requirements(
        SYLLABLES,
        UnitMap(_inventory(_unit("sh_plain", "sh", "plain"), _unit("sh_rounded", "sh", "rounded"))),
    )
    merged = build_requirements(
        SYLLABLES,
        UnitMap(_inventory(_unit("sh_shared", "sh", "plain", "rounded", decision="merge_supported"))),
    )

    finals = {item.final for item in SYLLABLES}
    assert len(split.vc) == len(base.vc) + len(finals)
    assert len(merged.vc) == len(base.vc)
    assert {("sha", "sh_plain"), ("shu", "sh_rounded"), ("shuo", "sh_rounded")} <= split.cv


def test_x_front_unrounded_context_resolves_from_plain_finals() -> None:
    units = UnitMap(
        _inventory(
            _unit("x_front_unrounded", "x", "front_unrounded"),
            _unit("x_rounded", "x", "rounded"),
        )
    )
    assert units.unit(_syllable("xi"), "internal") == "x_front_unrounded"
    assert units.unit(_syllable("xu"), "internal") == "x_rounded"
    assert not units.unmapped


def test_role_scoped_units_follow_position() -> None:
    inventory = _inventory(
        _unit("b_initial_i_series", "b", "initial:i_series"),
        _unit("b_internal_i_series", "b", "internal:i_series"),
        _unit("b_internal_other", "b", "internal:other"),
    )
    units = UnitMap(inventory)
    assert units.unit(_syllable("bi"), "initial") == "b_initial_i_series"
    assert units.unit(_syllable("bi"), "internal") == "b_internal_i_series"
    assert units.unit(_syllable("ba"), "initial") == "b_internal_other"


def test_unmatched_context_is_reported_not_guessed() -> None:
    units = UnitMap(_inventory(_unit("m_i", "m", "internal:i_series"), _unit("m_u", "m", "internal:u_series")))
    assert units.unit(_syllable("ma"), "internal") == "m"
    assert units.unmapped == {("m", "a")}


def test_reclist_covers_every_requirement_within_line_limit() -> None:
    units = UnitMap(
        _inventory(
            _unit("sh_plain", "sh", "plain"),
            _unit("sh_rounded", "sh", "rounded"),
            _unit("x_front_unrounded", "x", "front_unrounded"),
            _unit("x_rounded", "x", "rounded"),
        )
    )
    reclist = build_reclist(SYLLABLES, units, max_syllables=6, restarts=10, seed=1)

    assert not any(missing_requirements(reclist.lines, reclist.requirements, units).values())
    assert all(1 <= len(line) <= 6 for line in reclist.lines)
    assert len(reclist.lines) >= reclist.lower_bound


def test_reclist_is_deterministic_for_a_seed() -> None:
    units = UnitMap(None)
    first = build_reclist(SYLLABLES, units, restarts=5, seed=3).text()
    second = build_reclist(SYLLABLES, UnitMap(None), restarts=5, seed=3).text()
    assert first == second


def test_waived_vc_are_not_required() -> None:
    units = UnitMap(None)
    full = build_requirements(SYLLABLES, units)
    waived = frozenset({("a", "b"), ("i", "sh")})
    reduced = build_requirements(SYLLABLES, units, waived)
    assert reduced.vc == full.vc - waived
    assert lower_bound(reduced, 10) <= lower_bound(full, 10)


def test_line_limit_must_allow_a_connection() -> None:
    with pytest.raises(ValueError):
        build_reclist(SYLLABLES, UnitMap(None), max_syllables=1)


def test_inventory_yaml_round_trip(tmp_path: Path) -> None:
    inventory = _inventory(
        _unit("p_internal_i_series", "p", "internal:i_series", "internal:u_series"),
        _unit("sh_plain", "sh", "plain", decision="unresolved"),
    )
    path = tmp_path / "synthesis_inventory.yaml"
    path.write_text(synthesis_inventory_yaml(inventory), encoding="utf-8")
    assert load_synthesis_inventory(path) == inventory


def test_observed_syllables_and_cli(tmp_path: Path) -> None:
    voicebank = tmp_path / "bank"
    voicebank.mkdir()
    aliases = ["ba", "bi", "bu", "sha", "shu", "xi", "xu", "ya", "a"]
    (voicebank / "oto.ini").write_text(
        "".join(f"{alias}.wav={alias},0,0,0,0,0\n" for alias in aliases),
        encoding="utf-8",
    )
    syllables = observed_syllables(voicebank)
    texts = {item.text for item in syllables}
    assert {"ba", "bi", "sha", "xu", "ya"} <= texts
    assert next(item for item in syllables if item.text == "ya").onset == ""

    inventory = _inventory(_unit("sh_plain", "sh", "plain"), _unit("sh_rounded", "sh", "rounded"))
    inventory_path = tmp_path / "synthesis_inventory.yaml"
    inventory_path.write_text(synthesis_inventory_yaml(inventory), encoding="utf-8")
    output = tmp_path / "out" / "reclist.txt"

    status = reclist_main(
        [str(voicebank), "--inventory", str(inventory_path), "-o", str(output), "--restarts", "5"]
    )

    assert status == 0
    assert output.exists()
    assert output.with_suffix(".report.json").exists()
    assert all(len(line.split("_")) <= 10 for line in output.read_text(encoding="utf-8").splitlines())
