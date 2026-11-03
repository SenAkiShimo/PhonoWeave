"""Regression checks for literal, language-agnostic set-cover heuristic."""
from __future__ import annotations

import json
import pytest

from phonoweave.reclist import parse_reclist
from phonoweave.reclist_budget import plan_literal_budget
from phonoweave.reclist_budget_cli import main


def test_greedy_coverage_and_determinism():
    doc = parse_reclist("a b c\na b\nd e\na b c\n")
    result = plan_literal_budget(doc, 2)
    assert [x["source_line"] for x in result["selected_lines"]] == [1, 3]
    assert result["coverage"]["token_types"] == 5
    assert result["coverage"]["adjacent_pair_types"] == 3
    assert result["uncovered"] == {"tokens": [], "pairs": []}
    assert plan_literal_budget(doc, 2) == result


def test_budget_cap_zero_and_uncovered():
    doc = parse_reclist("a b\nc d\n")
    zero = plan_literal_budget(doc, 0)
    assert zero["selected_count"] == 0
    assert zero["uncovered"]["tokens"] == ["a", "b", "c", "d"]
    one = plan_literal_budget(doc, 1)
    assert one["selected_count"] == 1
    assert one["uncovered"]["tokens"] == ["c", "d"]


def test_duplicate_lines_do_not_spend_unused_budget():
    result = plan_literal_budget(parse_reclist("a b\na b\n"), 5)
    assert [x["source_line"] for x in result["selected_lines"]] == [1]
    assert result["unspent_line_budget"] == 4


def test_multilingual_literal_tokens_and_bom():
    doc = parse_reclist("\ufeff# header\nし しゃ\nʃ a\n", separator="whitespace")
    plan = plan_literal_budget(doc, 2)
    assert plan["selected_count"] == 2
    assert "ʃ" in plan["covered"]["tokens"]
    assert ["し", "しゃ"] in plan["covered"]["pairs"]
    assert plan["production_alias_export"] == "disabled"


def test_pairs_disabled_and_underscore_mode():
    doc = parse_reclist("a_b_c\na_c\n", separator="underscore")
    result = plan_literal_budget(doc, 1, include_pairs=False)
    assert result["selected_lines"][0]["tokens"] == ["a", "b", "c"]
    assert result["coverage"]["adjacent_pair_types_total"] == 0


@pytest.mark.parametrize("kwargs", [
    {"budget": -1}, {"budget": True}, {"budget": 1.2},
    {"budget": 1, "token_weight": 0},
    {"budget": 1, "pair_weight": False},
    {"budget": 1, "include_pairs": "yes"},
])
def test_invalid_inputs_fail(kwargs):
    with pytest.raises(ValueError):
        plan_literal_budget(parse_reclist("a b"), **kwargs)


def test_cli_creates_json_without_overwrite(tmp_path, capsys):
    source, output = tmp_path / "reclist.txt", tmp_path / "plan.json"
    source.write_text("a_b_c\na_b\n", encoding="utf-8")
    assert main([str(source), "--budget", "1", "--separator", "underscore",
                 "--output", str(output)]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["selected_lines"][0]["source_line"] == 1
    assert data["oto_timing_changes"] == "none"
    with pytest.raises(SystemExit) as exc:
        main([str(source), "--budget", "1", "--output", str(output)])
    assert exc.value.code == 2
