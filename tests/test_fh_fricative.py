from pathlib import Path

import pytest

from phonoweave.fh_fricative import _family, analyze_fh_fricative


def test_fh_family_mapping() -> None:
    for final in ("u", "ua", "uai", "uan", "uang", "ui", "un", "uo", "o", "ou", "ong"):
        assert _family(final) == "rounded"
    for final in ("a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng"):
        assert _family(final) == "other"


def test_fh_rejects_other_bases(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        analyze_fh_fricative(tmp_path, "s")
