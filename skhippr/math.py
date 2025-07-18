import numpy as np


def finite_differences(fun, kwargs, variable, h_step):

    x = kwargs[variable]

    try:
        n = len(x)
    except TypeError:
        n = 1

    f = fun(**kwargs)

    delta = x + h_step * np.eye(n)
    derivative = np.zeros((f.shape[0], n, *f.shape[1:]))
    for k in range(n):
        kwargs[variable] = np.squeeze(delta[k, :])
        derivative[:, k, ...] = (fun(**kwargs) - f) / h_step

    kwargs[variable] = x
    return derivative.squeeze()
