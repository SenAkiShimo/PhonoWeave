from phonoweave.diagnostic_selector import DiagnosticItem, DiagnosticSelection
from phonoweave.supplement_reclist import build_supplement_reclist


def _item(syllable: str, role: str | None, replicate: int = 1) -> DiagnosticItem:
    base = "zh" if syllable.startswith("zh") else syllable[0]
    return DiagnosticItem(
        base_unit=base,
        final=syllable[len(base):],
        syllable=syllable,
        context_family="test",
        role_scope=role,
        existing_observations=0,
        replicate=replicate,
    )


def test_internal_target_is_carried_in_internal_position() -> None:
    reclist = build_supplement_reclist(DiagnosticSelection(items=(_item("fa", "internal"),), unfilled=()))
    assert reclist.lines[0].text == "a_fa_a"


def test_unspecified_role_covers_initial_and_internal_in_one_line() -> None:
    reclist = build_supplement_reclist(DiagnosticSelection(items=(_item("zha", None),), unfilled=()))
    assert reclist.lines[0].text == "zha_a_zha_a"


def test_replicates_use_different_carriers() -> None:
    selection = DiagnosticSelection(
        items=(_item("ri", None, 1), _item("ri", None, 2)),
        unfilled=(),
    )
    reclist = build_supplement_reclist(selection)
    assert [line.text for line in reclist.lines] == ["ri_a_ri_a", "ri_e_ri_e"]
