from types import SimpleNamespace

import numpy as np

from phonoweave.stop_context import _holm_adjust, _stratum


def _sample(family: str, values: tuple[float, ...]):
    return SimpleNamespace(
        family=family,
        vector=lambda: np.array(values, dtype=np.float64),
    )


def test_stop_context_stratum_requires_two_per_family() -> None:
    samples = [
        _sample("i_series", (1, 1, 1, 1, 1)),
        _sample("u_series", (2, 2, 2, 2, 2)),
        _sample("u_series", (3, 3, 3, 3, 3)),
    ]
    assert _stratum(samples, "i_series", "u_series") is None


def test_stop_context_holm_is_monotone() -> None:
    adjusted = _holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]
