import numpy as np


def finite_differences(fun, kwargs, variable, h_step):

    x_orig = kwargs[variable]
    x = np.atleast_2d(x_orig)
    n = x.shape[0]
    if n == 1 and x.shape[1] > 1:
        x = x.T
        n = x.shape[0]

    f = fun(**kwargs)

    delta = h_step * np.eye(n)
    derivative = np.zeros((f.shape[0], n, *f.shape[1:]))
    for k in range(n):
        kwargs[variable] = np.squeeze(x + delta[:, [k]])
        derivative[:, k, ...] = (fun(**kwargs) - f) / h_step

    kwargs[variable] = x_orig
    return derivative.squeeze()
