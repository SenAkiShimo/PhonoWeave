from phonoweave.affricate import _SUPPORTED_BASES


def test_affricate_supported_bases() -> None:
    assert _SUPPORTED_BASES == {"zh", "ch", "z", "c"}
