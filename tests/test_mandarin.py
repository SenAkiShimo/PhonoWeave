from pathlib import Path

from phonoweave.mandarin import classify_alias, context_for, structure_for
from phonoweave.oto import OtoEntry


def _entry(alias: str) -> OtoEntry:
    path = Path("/tmp/oto.ini")
    return OtoEntry(path, Path("/tmp/a.wav"), alias, 0, 0, 0, 0, 0, 1)


def test_classify_retroflex_aliases() -> None:
    sh = classify_alias(_entry("- shi"))
    shu = classify_alias(_entry("sh u"))

    assert sh is not None
    assert sh.base_unit == "sh"
    assert sh.final == "i"
    assert context_for(sh.base_unit, sh.final) == "plain"

    assert shu is not None
    assert shu.base_unit == "sh"
    assert shu.final == "u"
    assert context_for(shu.base_unit, shu.final) == "rounded"


def test_classify_rhotic_contexts() -> None:
    ri = classify_alias(_entry("ri"))
    ru = classify_alias(_entry("ru"))

    assert ri is not None
    assert context_for(ri.base_unit, ri.final) == "front"

    assert ru is not None
    assert context_for(ru.base_unit, ru.final) == "rounded"


def test_classify_unanalyzed_mandarin_onsets_for_coverage() -> None:
    for alias, base, final in (
        ("ba", "b", "a"),
        ("- mao", "m", "ao"),
        ("tian", "t", "ian"),
        ("huang", "h", "uang"),
    ):
        observation = classify_alias(_entry(alias))
        assert observation is not None
        assert observation.base_unit == base
        assert observation.final == final


def test_structure_normalizes_zero_onset_orthography() -> None:
    for alias, final, initial in (
        ("ya", "ia", "y"),
        ("you", "iu", "y"),
        ("yue", "ve", "y"),
        ("wo", "uo", "w"),
        ("wei", "ui", "w"),
    ):
        observation = classify_alias(_entry(alias))
        assert observation is not None
        structure = structure_for(observation)
        assert structure.onset is None
        assert structure.final == final
        assert structure.orthographic_initial == initial


def test_structure_normalizes_palatal_u_to_umlaut_series() -> None:
    for alias, final in (
        ("ju", "v"),
        ("jue", "ve"),
        ("quan", "van"),
        ("xun", "vn"),
    ):
        observation = classify_alias(_entry(alias))
        assert observation is not None
        structure = structure_for(observation)
        assert structure.onset in {"j", "q", "x"}
        assert structure.final == final
