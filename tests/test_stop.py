from pathlib import Path
from types import SimpleNamespace

from phonoweave.stop import StopCandidate, _resolve_candidates


def _candidate(*, base: str, final: str, role: str, alias: str, start: float, end: float):
    entry = SimpleNamespace(
        wav_path=Path("/tmp/test.wav"),
        offset=start,
        preutterance=end - start,
        alias=alias,
    )
    observation = SimpleNamespace(base_unit=base, entry=entry)
    return StopCandidate(
        observation=observation,
        final=final,
        role=role,
        oto_set="A3",
    )


def test_stop_resolver_deduplicates_same_identity() -> None:
    rows = [
        _candidate(base="g", final="en", role="internal", alias="gen", start=10.0, end=50.0),
        _candidate(base="g", final="en", role="internal", alias="gen_alt", start=10.0, end=50.0),
    ]
    resolved, duplicate_removed, ambiguous_segments, ambiguous_observations = _resolve_candidates(rows)
    assert len(resolved) == 1
    assert duplicate_removed == 1
    assert ambiguous_segments == 0
    assert ambiguous_observations == 0


def test_stop_resolver_drops_conflicting_identity() -> None:
    rows = [
        _candidate(base="k", final="a", role="internal", alias="ka", start=10.0, end=50.0),
        _candidate(base="k", final="ai", role="internal", alias="kai", start=10.0, end=50.0),
    ]
    resolved, duplicate_removed, ambiguous_segments, ambiguous_observations = _resolve_candidates(rows)
    assert resolved == []
    assert duplicate_removed == 0
    assert ambiguous_segments == 1
    assert ambiguous_observations == 2
