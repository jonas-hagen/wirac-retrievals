import numpy as np


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

    def f_exp(x):
        return np.exp(-x)

    def f_lin(x):
        return 1.0 - (1.0 - np.exp(-1.0)) * x

    def f_gau(x):
        return np.exp(-x ** 2)

    if fname == 'lin':
        f = f_lin
    elif fname == 'exp':
        f = f_exp
    elif fname == 'gauss':
        f = f_gau
    else:
        raise NotImplementedError(f'Correlation type {fname} is not implemented, only "exp" and "lin" are valid.')

    x1, x2 = np.meshgrid(grid1, grid2)
    dist = np.abs(x1 - x2)

    x1, x2 = np.meshgrid(cl1, cl2)
    cl = (x1 + x2) / 2

    var = sigma1[:, np.newaxis].dot(sigma2[np.newaxis, :])

    S = var * f(dist / cl)

    if cutoff > 0:
        S[S < cutoff] = 0

    return S
