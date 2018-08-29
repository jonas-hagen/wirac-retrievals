import numpy as np
from retrievals import utils
import pytest


def test_exp_space():
    x = utils.exp_space(0, 100, 10, 40, 2)
    ref = np.array([0.000000, 31.540329, 37.871821, 39.517055, 39.968824,
                    40.082980, 40.709268, 42.928128, 51.918405, 100.000000])
    assert np.allclose(x, ref)


@pytest.mark.parametrize('N, M, a, center',
                         [(8, 100, 1, None),  # N, M, a, center
                          (300, 2**14, 1e-6, 2**13+2e3),
                          (300, 2**14, 1e-6, 2**13-2e3),
                          (301, 2**14+1, 1e-6, 2**13-2e3)])
def test_reduction_map(N, M, a, center):
    S = utils.reduction_map(N, M, a, center)

    # Shape is correct
    assert S.shape[1] == M
    if center is None:
        assert S.shape[0] == N

    # Row sum is one
    assert np.allclose(S.sum(axis=1), 1)

    # Columns are never 0 and exactly one row has a value
    assert np.all(np.sum(S > 0, axis=0) == 1)

    # Center is in the center :)
    if center is not None:
        col_sum = np.sum(S, axis=0)
        cs = np.nonzero(col_sum == 1)
        assert round(np.mean(cs)) == center
