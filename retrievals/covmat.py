import numpy as np
from scipy import sparse


def corr_fun(fname):
    """
    Define the functional form of correlations.

    The following types are available:
        "exp": f(x) = exp(-x)
        "lin": f(x) = 1.0 - x, for x > 1.0, 0.0 otherwise
        "gauss": f(x) = exp(-x^2)
    :param fname: Name of functional form.
    :return: A function 'f' taking one argument: corr = f(d/cl)
    """
    def f_exp(r):
        return np.exp(-r)

    def f_lin(r):
        return 1.0 - (1.0 - np.exp(-1.0)) * r

    def f_gauss(r):
        return np.exp(-r ** 2)

    if fname == 'lin':
        return f_lin
    elif fname == 'exp':
        return f_exp
    elif fname == 'gauss':
        return f_gauss
    else:
        raise NotImplementedError(
            f'Correlation type {fname} is not implemented, only "exp" and "lin" and "gauss" are valid.')


def covmat_1d(grid1, sigma1, cl1, grid2=None, sigma2=None, cl2=None, fname='lin', cutoff=0):
    """
    Creates a 1D covariance matrix for two retrieval quantities on given
    grids from a given functional form. Elements of the covariance matrix
    are computed as
        S_{i,j} = sigma_i * sigma_j * f(d_{i,j} / l_{i,j})
    where d_{i,j} is the distance between the two grid points and l_{i,j}
    the mean of the correlation lengths of the grid points.

    If a cutoff value is given elements with a correlation less than this
    are set to zero.

    :param grid1: The retrieval grid for the first retrieval quantity.
    :param sigma1: The variances of the first retrieval quantity.
    :param cl1: The correlations lengths of the first retrieval quantity.
    :param grid2: The retrieval grid for the second retrieval quantity. Default: grid1
    :param sigma2: The variances of the second retrieval quantity. Default: sigma1
    :param cl2: The correlations lengths of the second retrieval quantity. Default: cl1
    :param fname: Name of functional form of correlation, one of 'exp', 'lin', 'gauss'.
    :param cutoff: The cutoff value for covariance matrix elements.
    :return: The covariance matrix.
    """
    if grid2 is None:
        grid2 = grid1
        sigma2 = sigma1
        cl2 = cl1

    f = corr_fun(fname)

    x2, x1 = np.meshgrid(grid2, grid1)
    dist = np.abs(x2 - x1)

    x2, x1 = np.meshgrid(cl2, cl1)
    cl = (x2 + x1) / 2

    var = sigma1[:, np.newaxis] @ sigma2[np.newaxis, :]
    Sc = f(dist / cl)

    if cutoff > 0:
        Sc[Sc < cutoff] = 0
    S = var * Sc

    return S


def covmat_1d_sparse(grid1, sigma1, cl1, grid2=None, sigma2=None, cl2=None, fname='lin', cutoff=0):
    """
    Same as `covmat_1d` but creates and returns a sparse matrix.
    :param grid1:
    :param sigma1:
    :param cl1:
    :param grid2:
    :param sigma2:
    :param cl2:
    :param fname:
    :param cutoff:
    :return:
    """
    if grid2 is None:
        grid2 = grid1
        sigma2 = sigma1
        cl2 = cl1

    f = corr_fun(fname)

    n1 = len(grid1)
    n2 = len(grid2)
    row_ind = []
    col_ind = []
    values = []

    for i in range(n1):
        for j in range(n2):
            d = np.abs(grid1[i] - grid2[j])
            c = (cl1[i] + cl2[j]) / 2
            s = sigma1[i] * sigma2[j]
            value = f(d / c)
            if value >= cutoff:
                row_ind.append(i)
                col_ind.append(j)
                values.append(s * value)

    S = sparse.coo_matrix((values, (row_ind, col_ind)), shape=(n1, n2))

    return S


def covmat_3d(grid1, cl1, fname1,
              grid2, cl2, fname2,
              grid3, cl3, fname3,
              sigma, cutoff=0, separable=True):
    """
    Builds a correlation matrix for one retrieval species for a 3D retrieval grid.

    The correlation matrix is a 2 dimensional matrix with the 3 (spatial) dimensions stacked,
    where grid1 belongs to the fastest running dimension and grid3 to the slowest running dimension.

    :param grid1: Grid for first dimension.
    :param cl1: The correlations lengths for the first dimension, same shape as grid1.
    :param fname1: Name of functional form of correlation, one of 'exp', 'lin', 'gauss'.
    :param grid2: Grid for second dimension.
    :param cl2: ...
    :param fname2: ...
    :param grid3: Grid for third dimension.
    :param cl3: ...
    :param fname3: ...
    :param sigma: Variances for the retrieval quantity on 3d grid.
    :param cutoff: Correlation cut-off
    :param separable: Use separable or non-separable statistics.
                      If non-separable statistics are used, all correlation functions must be the same.
    :return: The correlation matrix with index of grid1 running fastest.
    """
    assert grid1.shape == cl1.shape, 'Dimension mismatch of grid1 and cl1.'
    assert grid2.shape == cl2.shape, 'Dimension mismatch of grid2 and cl2.'
    assert grid3.shape == cl3.shape, 'Dimension mismatch of grid3 and cl3.'
    assert sigma.shape == (len(grid1), len(grid2), len(grid3)), 'Dimension mismatch of sigma and grids.'

    if separable:
        # Use separable statistics
        # Total correlation is just the product of separate correlations
        f1 = corr_fun(fname1)
        f2 = corr_fun(fname2)
        f3 = corr_fun(fname3)

        def f(r1, r2, r3):
            return f1(r1) * f2(r2) * f3(r3)
    else:
        # Use non separable statistics
        # All dimensions must have same correlation function
        if not fname1 == fname2 == fname3:
            raise ValueError(f'For separable statistics, correlation function must be the same '
                             f'for all dimensions, got {fname1}, {fname2}, {fname3}.')
        f1 = corr_fun(fname1)

        def f(r1, r2, r3):
            return f1(np.sqrt(r1 ** 2 + r2 ** 2 + r3 ** 2))

    n1 = len(grid1)
    n2 = len(grid2)
    n3 = len(grid3)
    n = n1 * n2 * n3

    S = np.zeros((n, n))

    for idx1 in range(n):
        # index i belongs to grid1 and is the fastest running
        k1, j1, i1 = np.unravel_index(idx1, (n3, n2, n1))

        for idx2 in range(n):
            k2, j2, i2 = np.unravel_index(idx2, (n3, n2, n1))

            # determine distance and correlation for each grid separately
            d1 = np.abs(grid1[i1] - grid1[i2])
            c1 = (cl1[i1] + cl1[i2]) / 2

            d2 = np.abs(grid2[j1] - grid2[j2])
            c2 = (cl2[j1] + cl2[j2]) / 2

            d3 = np.abs(grid3[k1] - grid3[k2])
            c3 = (cl3[k1] + cl3[k2]) / 2

            # normalize distance with correlation length and compute total correlation
            value = f(d1 / c1, d2 / c2, d3 / c3)

            if value >= cutoff:
                var = sigma[i1, j1, k1] * sigma[i2, j2, k2]
                S[idx1, idx2] = var * value

    return S
