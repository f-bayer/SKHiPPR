import numpy as np


def finite_differences(fun, kwargs, variable, h_step):

    x = kwargs[variable]
    f = fun(**kwargs)

    delta = x + h_step * np.eye(len(x))
    derivative = np.zeros((len(f), len(x)))
    for k in range(len(x)):
        kwargs[variable] = delta[k, :]
        derivative[:, k] = (fun(**kwargs) - f) / h_step

    kwargs[variable] = x
    return derivative
