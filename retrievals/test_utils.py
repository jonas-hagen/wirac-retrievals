import numpy as np
from retrievals import utils

def test_exp_space():
    x = utils.exp_space(0, 100, 10, 40, 2)
    ref = np.array([0.000000, 31.540329, 37.871821, 39.517055, 39.968824,
                    40.082980, 40.709268, 42.928128, 51.918405, 100.000000])
    assert np.allclose(x, ref)
