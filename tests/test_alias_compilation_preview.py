"""Export preview contract: preserve source aliases and block collisions."""
import pytest
from phonoweave.alias_compilation_preview import compile_alias_preview


def row(alias, role, surface, left, right, pitch="A#3", origin="A#3/oto.ini", line=1):
    a = {"source_oto": origin, "line_number": line,
         "original_alias": alias, "candidate_unit_id": "sh_plain",
         "mapping_status": "candidate_review_only"}
    r = {"source_oto": origin, "line_number": line,
         "original_alias": alias, "role": role, "surface_alias": surface,
         "left_token": left, "right_token": right,
         "pitch_suffix": pitch, "pitch_suffix_status": "matched"}
    return a, r


def test_roundtrip_examples_and_cv_links_only():
    examples = [
        row("- shaA#3", "initial_cv", "- sha", "-", "sha", line=1),
        row("shaA#3", "standalone_cv", "sha", "", "sha", line=2),
        row("a shA#3", "vc_transition", "a sh", "a", "sh", line=3),
        row("ang erA#3", "vv_transition", "ang er", "ang", "er", line=4),
        row("_aiA#3", "prefixed_vowel", "_ai", "", "ai", line=5),
    ]
    records, summary = compile_alias_preview([a for a, _ in examples],
                                              [r for _, r in examples])
    assert summary["roundtrip_verified_rows"] == 5
    assert all(x["rendered_alias"] == x["original_alias"] for x in records)
    assert records[0]["candidate_unit_id"] == "sh_plain"
    assert records[2]["candidate_unit_id"] == ""


def test_cross_oto_alias_collision_detected():
    a, ra = row("shaA#3", "standalone_cv", "sha", "", "sha")
    b, rb = row("shaA#3", "standalone_cv", "sha", "", "sha",
                origin="A#3 EX/oto.ini")
    records, summary = compile_alias_preview([a,b], [ra,rb])
    assert summary["globally_colliding_aliases"] == 1
    assert summary["collision_affected_rows"] == 2
    assert all(x["compile_status"] == "global_alias_collision" for x in records)


def test_unsupported_and_mismatch_do_not_compile():
    a, ra = row("a xyA#3", "nonstandard_transition", "a xy", "a", "xy")
    b, rb = row("shaG#4", "standalone_cv", "shaG#4", "", "shaG#4", line=2)
    rb["pitch_suffix_status"] = "not_matched"
    rows, summary = compile_alias_preview([a,b], [ra,rb])
    assert [x["compile_status"] for x in rows] == [
        "unsupported_role", "unverified_pitch_suffix"]
    assert summary["roundtrip_verified_rows"] == 0


def test_source_alignment_is_required():
    a, ra = row("shaA#3", "standalone_cv", "sha", "", "sha")
    ra["line_number"] = 2
    with pytest.raises(ValueError, match="provenance"):
        compile_alias_preview([a], [ra])
