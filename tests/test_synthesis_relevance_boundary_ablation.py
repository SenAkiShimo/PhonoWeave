"""Guardrails for blinded donor-cut experimental staging."""
import json
from pathlib import Path
import pytest

from phonoweave.synthesis_relevance_boundary_ablation import (
    OFFSETS_MS, stage_boundary_ablation,
)

def test_three_oto_offsets_frozen():
    assert OFFSETS_MS == (-40.,-20.,0.)

def test_source_and_output_safety(tmp_path:Path):
    source=tmp_path/"source"
    source.mkdir()
    previous=tmp_path/"previous"
    previous.mkdir()
    output=tmp_path/"new"
    (previous/"experiment.json").write_text(json.dumps({"schema":"wrong"}))
    with pytest.raises(ValueError,match="Requires previous"):
        stage_boundary_ablation(source,previous,output)
    assert not output.exists()
    with pytest.raises(ValueError,match="within source"):
        stage_boundary_ablation(source,previous,source/"nested")
    with pytest.raises(FileExistsError):
        stage_boundary_ablation(source,previous,previous)

def test_blind_aliases_not_semitransparent():
    from phonoweave.synthesis_relevance_boundary_ablation import VERSION
    assert VERSION.endswith("v0.4")
