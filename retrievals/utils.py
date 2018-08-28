"""
Some utils, mostly ported code from matlab.
"""
import numpy as np


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
