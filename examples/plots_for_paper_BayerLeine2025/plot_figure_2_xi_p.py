"""
This script provides functions to compute and plot the recursively defined function xi_p(t) (cf. Bayer&Leine2025)
for a given sequence of integers p and frequency omega, thereby recreating Figure 2 of Bayer&Leine2025.

Functions:
----------
xi_p(ts: np.ndarray, p: list[int], omega: float = 1) -> np.ndarray
    Recursively computes the function xi_p(t) for a sequence p using the derivative property and numerical integration.
    Parameters:
        ts (np.ndarray): Array of time points, must start at 0.
        p (list[int]): Sequence of integers defining the scalar factor.
        omega (float, optional): Frequency parameter. Default is 1.
    Returns:
        np.ndarray: Computed values of xi_p(t) as a complex array.
xi_p_base_case(ts: np.ndarray, p: list[int], omega: float = 1) -> np.ndarray
    Computes the base case of xi_p(t) when p is a single integer (i.e., m = 1).
    Parameters:
        ts (np.ndarray): Array of time points.
        p (list[int]): List containing a single integer.
        omega (float, optional): Frequency parameter. Default is 1.
    Returns:
        np.ndarray: Computed values of xi_p(t) as a complex array.
plot_xi_p(p: list[int], omega: float = 1, n_periods: int = 2)
    Plots the real and imaginary parts of xi_p(t) over a specified number of periods and saves the plot as a TikZ file.
    Parameters:
        p (list[int]): Sequence of integers defining the recursion.
        omega (float, optional): Frequency parameter. Default is 1.
        n_periods (int, optional): Number of periods to plot. Default is 2.
Usage:
------
Run the script directly to generate and display plots for several example sequences p.

Dependencies:
-------------
- numpy
- matplotlib
- scipy.integrate
- tikzplotlib-patched

"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import tikzplotlib


def xi_p(ts: np.ndarray, p: list[int], omega: float = 1):
    """
    Recursively compute the function xi_p(t) using the derivative property.
    Parameters
    ----------
    ts : np.ndarray
        Array of time points at which to evaluate xi_p(t). The first element must be 0.
    p : list[int]
        List of integer indices specifying the order and frequencies for the recursion.
    omega : float, optional
        Angular frequency (default is 1).
    Returns
    -------
    np.ndarray
        Array of complex values representing xi_p(t) evaluated at each time in `ts`.
    Raises
    ------
    ValueError
        If the first time point in `ts` is not zero.
    Notes
    -----
    - For the base case (when `p` has length 1), the computation is delegated to `xi_p_base_case`.
    - For higher orders, the function is computed recursively using the derivative property
      and integrated numerically using the trapezoidal rule.
    """

    if len(p) == 1:
        return xi_p_base_case(ts, p, omega)
    elif ts[0] != 0:
        raise ValueError(f"Integration must start at t=0 but ts[0] is {ts[0]}")
    else:
        xi_p_dot = xi_p(ts, p[1:], omega) * np.exp(1j * p[0] * omega * ts)
        # integrate numerically using Simpson's rule
        return cumulative_trapezoid(xi_p_dot, x=ts, initial=0.0j)


def xi_p_base_case(ts: np.ndarray, p: list[int], omega: float = 1) -> np.ndarray:
    """
    Compute the base case of the xi_p function for a given time array and frequency index.
    Parameters
    ----------
    ts : np.ndarray
        Array of time values.
    p : list[int]
        List containing a single integer frequency index. The function is only valid if len(p) == 1.
    omega : float, optional
        Angular frequency (default is 1).
    Returns
    -------
    np.ndarray
        The computed base case values as a complex numpy array.
    Raises
    ------
    ValueError
        If the length of p is greater than 1.
    """

    if len(p) > 1:
        raise ValueError(
            f"This base case is only valid for len(p) = 1, but len(p) is {len(p)}"
        )

    if p[0] == 0:
        return ts.astype(complex)
    else:
        return 1 / (1j * p[0] * omega) * (np.exp(1j * p[0] * omega * ts) - 1)


def plot_xi_p(p, omega=1, n_periods=2):
    """
    Plots the real and imaginary parts of the function xi_p(t) over multiple periods.
    Parameters:
        p (array-like): Coefficients or parameters for the xi_p function.
        omega (float, optional): Angular frequency. Default is 1.
        n_periods (int, optional): Number of periods to plot. Default is 2.
    Notes:
        - The function generates a plot of the real and imaginary parts of xi_p(t) as functions of normalized time (t/T).
        - The plot is saved as a TikZ file in the 'plots' directory with a filename based on the input parameter p.
    """

    T = 2 * np.pi / omega
    ts = np.linspace(0, n_periods * T, n_periods * 250 + 1, endpoint=True)

    m = len(p)
    xi = xi_p(ts, p, omega)

    fig, ax = plt.subplots(1, 1)
    ax.plot(ts / T, np.real(xi))
    ax.plot(ts / T, np.imag(xi))
    # ax.plot(ts / T, np.abs(xi), ":")
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    # ax.plot(ts / T, ts**m / factorial(m), "--", color="gray")
    ax.set_title(f"p = {p}")
    ax.set_xlabel("t/T")
    ax.set_ylabel("xi_p(t)")
    tikzplotlib.save(
        f"plots/xi_p_{p}.tikz", axis_width="\\fwidth", axis_height="\\fheight"
    )


if __name__ == "__main__":
    plot_xi_p([3, -3, -8, 1, 2])
    plot_xi_p([3, -7, -6])
    plot_xi_p([6, 3, -3])
    plot_xi_p([-8, -4, -1])
    plot_xi_p([1, 0, 0])
    plt.show()
