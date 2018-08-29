"""
Some utils, mostly ported code from matlab.
"""
import numpy as np
from scipy import optimize


def reduction_map(num_rows, num_cols, a=1, center=None):
    """
    Create a data reduction matrix using geometric series to determine bin width
    :param num_rows: Number of data points that we want in the end.
    :param num_cols: Number of measurements.
    :param a: Irregularity parameter: 1 (more equally), 1e-9 more detail in the center
    :param center: Point with most detail
    :return: A matrix S that can be applied to data y_red = S*y
    """
    if center is None:
        center = num_cols//2

    left_width = center
    right_width = num_cols - left_width
    M = int(max(left_width, right_width))
    N = int(np.ceil(num_rows/2))

    # Find r
    def ser(r, i: int):
        return np.ceil(a * r**i)

    def seq(r, n: int):
        c = 0
        for i in range(n):
            c += ser(r, i)
        return c

    # Find sign changing interval
    upper = int(2)
    while seq(upper, N) < M:
        upper = upper * 2

    r = optimize.brentq(lambda x: seq(x, N)-M, 1+1e-9, upper)

    # Fill S
    S = np.zeros((N, M))
    start = int(seq(r, 0))
    for i in range(N-1):
        stop = int(seq(r, i+1))
        S[i, start:stop] = 1
        start = stop
    S[i+1, stop:] = 1

    # Construct full matrix
    St = np.flipud(np.fliplr(S))
    z = np.zeros_like(S)
    Sfull = np.block([[St, z],
                      [z, S]])

    # If necessary, crop
    if left_width > right_width:
        # Crop right
        Sfull = Sfull[:, :num_cols]
    else:
        # Crop left
        Sfull = Sfull[:, -num_cols:]
    Sfull = Sfull[Sfull.sum(axis=1) > 0, :]

    # Normalize
    row_sums = Sfull.sum(axis=1)
    Snorm = Sfull / row_sums[:, np.newaxis]

    return Snorm


def exp_space(a, b, num, c=None, r=2):
    """
    Gives N points in (a,b), with higher concentration around c.
    :param a: Start value
    :param b: End value
    :param num: Number of points
    :param c: Center (default: (a+b)/2)
    :param r: Irregularity parameter (default: 2)
    :return: Array length N.
    """

    def f(x):
        y = np.exp(x**2) - 1
        neg = x < 0
        if np.isscalar(x):
            y = -y if neg else y
        else:
            y[neg] = -y[neg]
        return y

    def finv(y):
        neg = y < 0
        y = np.abs(y)
        x = np.sqrt(np.log(y+1))
        if np.isscalar(y):
            x = -x if neg else x
        else:
            x[neg] = -x[neg]
        return x

    def scale(y, a, b):
        s = (b-a)*(y-y[0])/(y[-1]-y[0]) + a
        return s

    if c is None:
        c= (a+b)/2

    len_left = c-a  # length left from center
    len_right = b-c  # length right from center

    # The function f(x)=exp(x^2) (defined above) shall be evaluated
    # between 0 and maximally R.
    yres = f(r) / max([len_left, len_right])

    x_left = finv(-len_left*yres)  # Left bound on x
    x_right = finv( len_right*yres)  # Right bound on x

    y = f(np.linspace(x_left, x_right, num))
    return scale(y, a, b)
