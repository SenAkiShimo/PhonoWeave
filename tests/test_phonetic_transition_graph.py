"""Language-neutral symbolic transition coverage; no actual acoustic labels inferred."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from phonoweave.reclist import parse_reclist
from phonoweave.phonetic_transition_graph import (
    LEXICON_SCHEMA, SCHEMA, compile_transition_budget,
)
from phonoweave.phonetic_transition_cli import main


def phone(symbol, role, grapheme):
    return {"symbol": symbol, "role": role, "grapheme": grapheme}


def lexicon():
    return {
        "schema": LEXICON_SCHEMA,
        "language": "zh-CN-pinyin-manual-example",
        "entries": {
            "shi": [phone("ʂ", "C", "sh"), phone("ɻ̩", "V", "i")],
            "li": [phone("l", "C", "l"), phone("i", "V", "i")],
            "ma": [phone("m", "C", "m"), phone("a", "V", "a")],
            "sh": [phone("ʂ", "C", "sh"), phone("ɻ̩", "V", "i")],
        },
    }


def test_same_spelling_different_nuclei_and_vc():
    document = parse_reclist("shi ma\nli ma\n")
    report = compile_transition_budget(document, lexicon(), 2)
    assert report["schema"] == SCHEMA
    assert report["phonetic_policy"]["selected_count"] == 2
    ambiguity = next(x for x in report["spelling_realization_ambiguities"]
                     if x["grapheme"] == "i")
    assert {x["symbol"] for x in ambiguity["realizations"]} == {"i", "ɻ̩"}
    crossings = [x for row in report["phonetic_policy"]["selected_lines"]
                 for x in row["new_features"]
                 if x["kind"] == "transition" and x["phase"] == "across_syllables"]
    assert {x["left"] for x in crossings} == {"i", "ɻ̩"}
    assert all(x["pattern"] == "VC" and x["right"] == "m" for x in crossings)
    assert report["phonetic_policy"]["coverage"]["transition_types"]["across_syllables.VC"] == {
        "covered": 2, "total": 2,
    }
    assert report["production_ready"] is False
    assert compile_transition_budget(document, lexicon(), 2) == report


def test_cc_and_vv_across_syllable_boundary():
    entries = {
        "a": [phone("a", "V", "a")],
        "ai": [phone("a", "V", "a"), phone("i", "V", "i")],
        "sta": [phone("s", "C", "s"), phone("t", "C", "t"), phone("a", "V", "a")],
        "tra": [phone("t", "C", "t"), phone("r", "C", "r"), phone("a", "V", "a")],
        "ats": [phone("a", "V", "a"), phone("t", "C", "t"), phone("s", "C", "s")],
    }
    lex = {"schema": LEXICON_SCHEMA, "language": "constructed-test", "entries": entries}
    report = compile_transition_budget(parse_reclist("a ai\nats tra\nsta\n"), lex, 3)
    types = report["phonetic_policy"]["coverage"]["transition_types"]
    assert types["across_syllables.VV"] == {"covered": 1, "total": 1}
    assert types["across_syllables.CC"] == {"covered": 1, "total": 1}
    assert types["within_syllable.CC"]["total"] >= 1
    assert types["within_syllable.CV"]["total"] >= 1


def test_one_budget_equal_prefix_and_literal_comparison():
    doc = parse_reclist("shi ma\nli ma\nshi li\n")
    result = compile_transition_budget(doc, lexicon(), 1)
    assert len(result["matched_prefix_curve"]) == 2
    for row in result["matched_prefix_curve"]:
        assert row["phonetic_policy"]["distinct_phone_types"] <= row["phonetic_policy"]["distinct_phone_types_total"]
        assert row["literal_policy"]["distinct_phone_types"] <= row["literal_policy"]["distinct_phone_types_total"]
    assert result["settings"]["cost_unit"] == "recording_line_not_measured_duration"


def test_unmapped_tokens_fail_closed():
    with pytest.raises(ValueError, match="missing explicit pronunciations"):
        compile_transition_budget(parse_reclist("shi UNKNOWN"), lexicon(), 2)


def test_invalid_mapping_rejected():
    bad = deepcopy(lexicon())
    bad["entries"]["li"] = [phone("l", "C", "l")]
    with pytest.raises(ValueError, match="no syllabic nucleus"):
        compile_transition_budget(parse_reclist("li"), bad, 1)
    bad = deepcopy(lexicon())
    bad["entries"]["li"] = [phone("l", "C", "l"), phone("i", "Q", "i")]
    with pytest.raises(ValueError, match="invalid pronunciation segment"):
        compile_transition_budget(parse_reclist("li"), bad, 1)


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_budget(budget):
    with pytest.raises(ValueError, match="nonnegative integer"):
        compile_transition_budget(parse_reclist("shi"), lexicon(), budget)


def test_comparison_no_abstention_fabrication():
    result = compile_transition_budget(parse_reclist("shi ma\nli ma"), lexicon(), 0)
    assert result["phonetic_policy"]["selected_count"] == 0
    assert result["literal_policy"]["selected_count"] == 0
    assert len(result["matched_prefix_curve"]) == 1
    assert result["phonetic_policy"]["coverage"]["weighted_covered"] == 0


def test_cli_writes_only_new_output(tmp_path):
    reclist, lex, out = tmp_path/"list.txt", tmp_path/"lex.json", tmp_path/"out.json"
    reclist.write_text("shi ma\nli ma\n", encoding="utf-8")
    lex.write_text(json.dumps(lexicon(), ensure_ascii=False), encoding="utf-8")
    assert main([str(reclist), "--lexicon", str(lex), "--budget", "2",
                 "--output", str(out)]) == 0
    assert json.loads(out.read_text())["schema"] == SCHEMA
    with pytest.raises(SystemExit) as exc:
        main([str(reclist), "--lexicon", str(lex), "--budget", "2",
              "--output", str(out)])
    assert exc.value.code == 2


def test_cli_duplicate_json_key_rejected(tmp_path):
    reclist, lex, out = tmp_path/"list.txt", tmp_path/"lex.json", tmp_path/"out.json"
    reclist.write_text("shi\n", encoding="utf-8")
    lex.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main([str(reclist), "--lexicon", str(lex), "--budget", "1",
              "--output", str(out)])
    assert exc.value.code == 2
    assert not out.exists()
