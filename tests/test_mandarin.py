from pathlib import Path

from phonoweave.mandarin import classify_alias, context_for
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
