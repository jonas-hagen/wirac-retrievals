import numpy as np
from scipy import sparse


def corr_fun(fname):
    def f_exp(x):
        return np.exp(-x)

    def f_lin(x):
        return 1.0 - (1.0 - np.exp(-1.0)) * x

    def f_gauss(x):
        return np.exp(-x ** 2)

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
    grids from a given functional form. Elements  of the covariance matrix
    are computed as
        S_{i,j} = sigma_i * sigma_j * f(d_{i,j} / l_{i,j})
    where d_{i,j} is the distance between the two grid points and l_{i,j}
    the mean of the correlation lengths of the grid points.

    If a cutoff value co is given elements with absolute value less than this
    are set to zero.

    The following functional forms are available:
        "exp": f(x) = exp(-x)
        "lin": f(x) = 1.0 - x, for x > 1.0, 0.0 otherwise
        "gauss": f(x) = exp(-x^2)

    Based on implementations in the ARTS and Atmlab projects by Patrick Eriksson and Simon Pfreundschuh.

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
    S = var * f(dist / cl)

    if cutoff > 0:
        S[S < cutoff] = 0

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
            value = s * f(d/c)
            if value >= cutoff:
                row_ind.append(i)
                col_ind.append(j)
                values.append(value)

    S = sparse.coo_matrix((values, (row_ind, col_ind)), shape=(n1, n2))

    return S
