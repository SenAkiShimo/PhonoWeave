from __future__ import annotations

import json
from pathlib import Path

from phonoweave import plan_cli
from phonoweave.inventory import InventoryDecision, VoicebankInventoryAnalysis


def _voicebank(root: Path) -> Path:
    bank = root / "bank"
    bank.mkdir()
    aliases = ["ba", "bi", "bu", "sha", "shu", "she", "zha", "zhu", "zhe", "xi", "xu", "ma", "a"]
    (bank / "oto.ini").write_text(
        "".join(f"{alias}.wav={alias},0,0,0,0,0\n" for alias in aliases),
        encoding="utf-8",
    )
    return bank


def test_plan_runs_one_analysis_and_writes_every_output(monkeypatch, tmp_path, capsys) -> None:
    bank = _voicebank(tmp_path)
    calls = []

    def fake_analysis(root, single_layer=False):
        calls.append(root)
        return VoicebankInventoryAnalysis(
            voicebank=root,
            decisions=[
                InventoryDecision(
                    base_unit="sh",
                    class_name="fricative",
                    acoustic_evidence="strongly_supported",
                    synthesis_evidence="supported_under_proxy",
                    decision="split_recommended",
                    confidence="high",
                    notes=(),
                ),
                InventoryDecision(
                    base_unit="zh",
                    class_name="affricate",
                    acoustic_evidence="weak_or_inconsistent",
                    synthesis_evidence="not_tested",
                    decision="unresolved",
                    confidence="low",
                    notes=(),
                ),
            ],
        )

    monkeypatch.setattr(plan_cli, "analyze_voicebank_inventory", fake_analysis)
    output = tmp_path / "out"

    status = plan_cli.main(
        [str(bank), "-o", str(output), "--restarts", "5", "--with-supplement"]
    )

    assert status == 0
    assert len(calls) == 1
    for name in (
        "synthesis_inventory.yaml",
        "reclist.txt",
        "reclist.report.json",
        "bank_audit.txt",
        "bank_audit.json",
        "supplement_reclist.txt",
        "evidence_gaps.json",
    ):
        assert (output / name).exists()

    report = json.loads((output / "reclist.report.json").read_text(encoding="utf-8"))
    assert "sh_plain" in report["units"] and "sh_rounded" in report["units"]
    assert report["collapsed_unresolved"] == ["zh"]

    gaps = json.loads((output / "evidence_gaps.json").read_text(encoding="utf-8"))
    assert [gap["base_unit"] for gap in gaps["gaps"]] == ["zh"]
    assert (output / "supplement_reclist.txt").read_text(encoding="utf-8").strip()
    assert "Supplement reclist lines" in capsys.readouterr().out


def test_plan_skips_supplement_by_default(monkeypatch, tmp_path, capsys) -> None:
    bank = _voicebank(tmp_path)
    monkeypatch.setattr(
        plan_cli,
        "analyze_voicebank_inventory",
        lambda root, single_layer=False: VoicebankInventoryAnalysis(voicebank=root, decisions=[]),
    )
    output = tmp_path / "out"

    assert plan_cli.main([str(bank), "-o", str(output), "--restarts", "5"]) == 0

    assert (output / "reclist.txt").exists()
    assert (output / "bank_audit.txt").exists()
    assert not (output / "supplement_reclist.txt").exists()
    assert not (output / "evidence_gaps.json").exists()
    assert "no recording needed" in capsys.readouterr().out


def _unit(base, context, decision, confidence):
    from phonoweave.synthesis_inventory import SynthesisUnit

    return SynthesisUnit(
        id=f"{base}_{context}",
        base_unit=base,
        contexts=(context,),
        source_group=f"{base}_{context}",
        canonical_context=None,
        decision=decision,
        confidence=confidence,
        acoustic_evidence="supported",
        synthesis_evidence="supported_under_proxy",
    )


def test_merge_keeps_a_split_any_analysis_calls() -> None:
    from phonoweave.synthesis_inventory import SynthesisInventory, merge_inventories

    def inventory(units):
        return SynthesisInventory("s", "mandarin", Path("x"), tuple(units))

    pooled = inventory([
        _unit("x", "plain", "split_recommended", "moderate"),
        _unit("x", "rounded", "split_recommended", "moderate"),
        _unit("sh", "plain", "unresolved", "low"),
        _unit("sh", "rounded", "unresolved", "low"),
    ])
    single = inventory([
        _unit("sh", "plain", "split_recommended", "low"),
        _unit("sh", "rounded", "split_recommended", "low"),
        _unit("x", "plain", "unresolved", "low"),
        _unit("x", "rounded", "unresolved", "low"),
    ])

    merged = merge_inventories([pooled, single])

    decisions = {(u.base_unit, u.decision) for u in merged.units}
    assert decisions == {("x", "split_recommended"), ("sh", "split_recommended")}
    assert len(merged.units) == 4


def test_plan_uses_given_inventories_without_analysing(monkeypatch, tmp_path, capsys) -> None:
    from phonoweave.synthesis_inventory import SynthesisInventory, synthesis_inventory_yaml

    bank = _voicebank(tmp_path)
    given = tmp_path / "given.yaml"
    given.write_text(
        synthesis_inventory_yaml(
            SynthesisInventory(
                "s",
                "mandarin",
                Path("elsewhere"),
                (
                    _unit("sh", "plain", "split_recommended", "moderate"),
                    _unit("sh", "rounded", "split_recommended", "moderate"),
                ),
            )
        ),
        encoding="utf-8",
    )

    def refuse(*args, **kwargs):
        raise AssertionError("the voicebank must not be analysed")

    monkeypatch.setattr(plan_cli, "analyze_voicebank_inventory", refuse)
    output = tmp_path / "out"

    status = plan_cli.main(
        [str(bank), "-o", str(output), "--restarts", "5", "--inventory", str(given)]
    )

    assert status == 0
    report = json.loads((output / "reclist.report.json").read_text(encoding="utf-8"))
    assert "sh_plain" in report["units"] and "sh_rounded" in report["units"]
    assert "inventory given.yaml" in (output / "summary.txt").read_text(encoding="utf-8")
    assert "Unresolved decisions: 0" in capsys.readouterr().out
