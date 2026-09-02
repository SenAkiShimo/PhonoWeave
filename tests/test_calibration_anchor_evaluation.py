from __future__ import annotations

from pathlib import Path

from phonoweave.calibration_anchor_evaluation import evaluate_manual_anchors


def _write_labels(session: Path) -> None:
    analysis = session / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "calibration_manual_anchor_labels_v0.2.tsv").write_text(
        "# prompt_index\toccurrence\tanchor_ms_after_cue\tstatus\n"
        "3\t1\t100.0\tok\n"
        "3\t2\t130.0\tok\n"
        "7\t1\t200.0\tok\n"
        "7\t2\t240.0\tuncertain\n",
        encoding="utf-8",
    )


def test_manual_evaluation_metrics_and_repeat_lag_sign(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session"
    _write_labels(session)

    paired = {
        "anchors": [
            {
                "prompt_index": 3,
                "base_unit": "ch",
                "class_name": "affricate",
                "context_family": "rounded",
                "syllable": "chua",
                "occurrence_1_anchor_ms_after_cue": 120.0,
                "occurrence_2_anchor_ms_after_cue": 145.0,
                "qc": "usable",
            },
            {
                "prompt_index": 7,
                "base_unit": "x",
                "class_name": "fricative",
                "context_family": "plain",
                "syllable": "xi",
                "occurrence_1_anchor_ms_after_cue": 170.0,
                "occurrence_2_anchor_ms_after_cue": 260.0,
                "qc": "repeat_anchor_review",
            },
        ]
    }
    early = {
        "alignments": [
            {
                "prompt_index": 3,
                "base_unit": "ch",
                "class_name": "affricate",
                "context_family": "rounded",
                "syllable": "chua",
                "consensus_lag_ms": 20.0,
                "qc": "usable",
            },
            {
                "prompt_index": 7,
                "base_unit": "x",
                "class_name": "fricative",
                "context_family": "plain",
                "syllable": "xi",
                "consensus_lag_ms": 50.0,
                "qc": "usable",
            },
        ]
    }

    monkeypatch.setattr(
        "phonoweave.calibration_anchor_evaluation.analyze_paired_anchors",
        lambda _: paired,
    )
    monkeypatch.setattr(
        "phonoweave.calibration_anchor_evaluation.analyze_early_repeat_alignment",
        lambda _: early,
    )

    payload = evaluate_manual_anchors(session)

    absolute = payload["absolute_anchor"]
    metrics = absolute["primary_metrics"]
    # Primary absolute rows are 3/1 (+20), 3/2 (+15), 7/1 (-30).
    # 7/2 is uncertain and must not affect primary metrics.
    assert metrics["n"] == 3
    assert metrics["mae_ms"] == 21.667
    assert metrics["median_signed_error_ms"] == 15.0
    assert len(absolute["uncertain_rows"]) == 1

    relative = payload["relative_repeat_alignment"]["primary_metrics"]
    # Manual lag for prompt 3 is occurrence2-occurrence1 = +30 ms.
    # Auto lag is +20 ms, therefore signed error must be -10 ms.
    # Prompt 7 is excluded because occurrence 2 is uncertain.
    assert relative["n"] == 1
    assert relative["mae_ms"] == 10.0
    assert relative["median_signed_error_ms"] == -10.0

    assert (session / "analysis" / "calibration_anchor_evaluation_v0.4.json").is_file()
