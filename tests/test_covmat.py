import pytest
import numpy as np
from retrievals import covmat


def test_covmat_1d():
    reference = np.array(
        [0.090000, 0.054588, 0.033109, 0.020082, 0.012180, 0.007388, 0.004481, 0.002718, 0.000000, 0.000000, 0.054588,
         0.090000, 0.054588, 0.033109, 0.020082, 0.012180, 0.007388, 0.004481, 0.002718, 0.000000, 0.033109, 0.054588,
         0.090000, 0.054588, 0.033109, 0.020082, 0.012180, 0.007388, 0.004481, 0.002718, 0.020082, 0.033109, 0.054588,
         0.090000, 0.054588, 0.033109, 0.020082, 0.012180, 0.007388, 0.004481, 0.012180, 0.020082, 0.033109, 0.054588,
         0.090000, 0.054588, 0.033109, 0.020082, 0.012180, 0.007388, 0.007388, 0.012180, 0.020082, 0.033109, 0.054588,
         0.090000, 0.054588, 0.033109, 0.020082, 0.012180, 0.004481, 0.007388, 0.012180, 0.020082, 0.033109, 0.054588,
         0.090000, 0.054588, 0.033109, 0.020082, 0.002718, 0.004481, 0.007388, 0.012180, 0.020082, 0.033109, 0.054588,
         0.090000, 0.054588, 0.033109, 0.000000, 0.002718, 0.004481, 0.007388, 0.012180, 0.020082, 0.033109, 0.054588,
         0.090000, 0.054588, 0.000000, 0.000000, 0.002718, 0.004481, 0.007388, 0.012180, 0.020082, 0.033109, 0.054588,
         0.090000]).reshape((10, 10))

    x = np.arange(10)
    sigma = 0.3 * np.ones_like(x)
    cl = 2 * np.ones_like(x)
    cm = covmat.covmat_1d(x, sigma, cl, fname='exp', cutoff=0.02)

    assert np.allclose(cm, reference, atol=0.0001)


def test_covmat_1d_2():
    reference = np.array([[0.09, 0.05458776, 0.03310915, 0.02008171, 0.],
                          [0.05458776, 0.09, 0.05458776, 0.03310915, 0.02008171],
                          [0.03310915, 0.05458776, 0.09, 0.05458776, 0.03310915],
                          [0.02008171, 0.03310915, 0.05458776, 0.09, 0.05458776],
                          [0., 0.02008171, 0.03310915, 0.05458776, 0.09]])

    x1 = np.arange(5)
    sigma1 = 0.3 * np.ones_like(x1)
    cl1 = 2 * np.ones_like(x1)
    cm = covmat.covmat_1d(x1, sigma1, cl1, fname='exp', cutoff=0.02/(0.3**2))

    assert np.allclose(cm, reference)


def test_covmat_1d_sparse():
    x1 = np.arange(5)
    sigma1 = 0.3 * np.ones_like(x1)
    cl1 = 2 * np.ones_like(x1)

    S1 = covmat.covmat_1d(x1, sigma1, cl1, fname='lin', cutoff=0.02)
    S2 = covmat.covmat_1d_sparse(x1, sigma1, cl1, fname='lin', cutoff=0.02)

    assert np.allclose(S1, S2.toarray())


def test_covmat_1d_sparse_2():
    x1 = np.arange(5)
    sigma1 = 0.3 * np.ones_like(x1)
    cl1 = 2 * np.ones_like(x1)
    x2 = np.arange(10)
    sigma2 = 0.3 * np.ones_like(x2)
    cl2 = 2 * np.ones_like(x2)

    S1 = covmat.covmat_1d(x1, sigma1, cl1, x2, sigma2, cl2, fname='lin', cutoff=0.02)
    S2 = covmat.covmat_1d_sparse(x1, sigma1, cl1, x2, sigma2, cl2, fname='lin', cutoff=0.02)

    assert np.allclose(S1, S2.toarray())


def test_covmat_3d():
    n1 = 3
    n2 = 4
    n3 = 5

    x1 = np.arange(n1)
    cl1 = 2 * np.ones_like(x1)
    x2 = np.arange(n2)
    cl2 = 2 * np.ones_like(x2)
    x3 = np.arange(n3)
    cl3 = 2 * np.ones_like(x3)
    sigma = np.ones((n1, n2, n3))

    S = covmat.covmat_3d(x1, cl1, 'lin',
                         x2, cl2, 'lin',
                         x3, cl3, 'lin',
                         sigma)

    # Check a few values known from Atmlab implementation
    tol = 0.0001
    assert np.isclose(S[4, 4], 1, atol=tol)
    assert np.isclose(S[4, 7], 0.683939720585721, atol=tol)
    assert np.isclose(S[19, 39], 0.172084269003133, atol=tol)
    assert np.isclose(S[23, 54], 0.013038082601344, atol=tol)