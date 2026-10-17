from pathlib import Path

from phonoweave.alias_role_preview import classify_alias_role


def test_core_pitch_aliased_roles():
    oto = Path("/bank/A#3/oto.ini")
    examples = {
        "- shaA#3": ("initial_cv", "-", "sha"),
        "shaA#3": ("standalone_cv", "", "sha"),
        "a shA#3": ("vc_transition", "a", "sh"),
        "ang erA#3": ("vv_transition", "ang", "er"),
        "_aiA#3": ("prefixed_vowel", "", "ai"),
        "aiA#3": ("standalone_vowel", "", "ai"),
        "a 吸RA#3": ("modified_or_special_transition", "a", "吸R"),
        "a xyA#3": ("nonstandard_transition", "a", "xy"),
        "A#3": ("empty_or_pitch_only", "", ""),
    }
    for original, expected in examples.items():
        parsed = classify_alias_role(original, oto)
        assert (parsed.role, parsed.left_token, parsed.right_token) == expected
        assert parsed.pitch_suffix_status == "matched"


def test_unsupported_and_mismatched_suffix_stays_unmodified():
    invalid = classify_alias_role("a shG#4", Path("/bank/A#3/oto.ini"))
    assert invalid.pitch_suffix_status == "not_matched"
    assert invalid.surface == "a shG#4"
    assert invalid.review_status == "manual_review"
    breath = classify_alias_role("br1", Path("/bank/breathe/oto.ini"))
    assert breath.role == "special_or_unknown"
    assert breath.pitch_suffix_status == "not_applicable"


def test_nonstandard_tokens_are_not_automatically_mapped():
    item = classify_alias_role("ing shwD#4", Path("/bank/D#4/oto.ini"))
    assert item.role == "nonstandard_transition"
    assert item.review_status == "manual_review"
