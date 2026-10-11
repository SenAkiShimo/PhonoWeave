import numpy as np
from phonoweave.calibration_noise_floor_contrast import power_excess_db

def test_power_excess_is_zero_at_noise_floor():
    out=power_excess_db(np.array([-70.,-60.]),-60.)
    assert out[0]<-100 and out[1]<-100

def test_power_excess_above_floor_is_finite_and_monotone():
    out=power_excess_db(np.array([-60.,-50.,-40.]),-70.)
    assert np.isfinite(out).all()
    assert np.all(np.diff(out)>0)
