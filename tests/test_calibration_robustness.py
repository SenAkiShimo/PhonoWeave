from __future__ import annotations
import numpy as np
import pytest
from phonoweave.calibration_robustness import perturb, SCENARIOS

def test_deterministic_noise_perturbation():
    a=np.array([0.0,.01,.02,-.3,.2]*100)
    b=perturb(a,"noise_20db_snr",np.random.default_rng(42))
    c=perturb(a,"noise_20db_snr",np.random.default_rng(42))
    assert np.array_equal(b,c)
    assert not np.array_equal(b,a)

def test_gain_is_exact_scaling():
    a=np.array([.02,-.3,0.])
    assert np.allclose(perturb(a,"gain_minus_12db",np.random.default_rng(2)),a*10**(-12/20))
    assert np.allclose(perturb(a,"gain_plus_6db",np.random.default_rng(2)),a*10**(6/20))

def test_all_scenarios_finite():
    rng=np.random.default_rng(1)
    y=rng.normal(0,.03,1600)
    for s in SCENARIOS:
        z=perturb(y,s,np.random.default_rng(3))
        assert z.shape==y.shape and np.all(np.isfinite(z))
