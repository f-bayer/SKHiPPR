import numpy as np


def duffing(
    t: float,
    x: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    F: float,
    delta: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:

    if x.shape[0] != 2:
        raise ValueError("first dimension of x must have length n_dof=2")

    f = np.zeros_like(x)
    f[0, ...] = x[1, ...]
    f[1, ...] = (
        -alpha * x[0, ...]
        - delta * x[1, ...]
        - beta * x[0, ...] ** 3
        + F * np.cos(omega * t)
    )

    df_dx = np.zeros((2, 2, *x.shape[1:]), dtype=x.dtype)
    df_dx[0, 1, ...] = 1
    df_dx[1, 0, ...] = -alpha - 3 * beta * x[0, ...] ** 2
    df_dx[1, 1, ...] = -delta

    df_dF = np.zeros_like(x)
    df_dF[1, ...] = np.cos(omega * t)

    df_dom = np.zeros_like(x)
    df_dom[1, ...] = -F * t * np.sin(omega * t)

    derivatives = dict()
    derivatives["x"] = df_dx
    derivatives["F"] = df_dF
    # derivatives["omega"] = df_dom # -- CAUTION This breaks HBM because cos(omega*t) does not change in HBM...
    derivatives["T"] = -df_dom * (omega**2 / (2 * np.pi))

    return f, derivatives
