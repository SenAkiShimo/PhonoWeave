from phonoweave.source_audit import SourceIdentityLabel, _identity_status


def _label(final: str, role: str) -> SourceIdentityLabel:
    return SourceIdentityLabel(
        base_unit="g",
        final=final,
        role=role,
        alias="ge",
        oto_set="A3",
    )


def test_same_identity_is_duplicate() -> None:
    labels = (_label("en", "internal"), _label("en", "internal"))
    assert _identity_status(labels) == "duplicate"


def test_role_conflict_is_ambiguous() -> None:
    labels = (_label("en", "initial"), _label("en", "internal"))
    assert _identity_status(labels) == "ambiguous"


def test_final_conflict_is_ambiguous() -> None:
    labels = (_label("en", "internal"), _label("eng", "internal"))
    assert _identity_status(labels) == "ambiguous"
