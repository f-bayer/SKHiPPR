import numpy as np

"""
This script analyzes and visualizes decay of Fourier coefficients of the Jacobian matrix
within the Harmonic Balance Method (HBM) framework, specifically for the Duffing oscillator system.
This recreates Figure 9 from Bayer&Leine2025.

Main Features:
--------------
- Solves for periodic solutions of the Duffing oscillator using HBM.
- Identifies exponential decay parameters (a, b) from the solution's spectral content.
- Visualizes the solution in phase space and plots the spectral norm of the Jacobian's Fourier coefficients.
- Compares the nonlinear Duffing oscillator with its linearized counterpart (beta=0).

Functions:
----------
- main(params_duffing, fourier, x_init=None, idx_k=None):
    Solves the Duffing oscillator for given parameters, identifies exponential decay parameters, and generates relevant plots.
- determine_J_norms(prb, threshold=1e-15):
    Computes the 2-norms of the Jacobian's Fourier coefficients for a given HBMProblem.
- plot_bar(x_vals, y_vals, threshold=0, ax=None):
    Plots a logarithmic bar chart of spectral norms with a threshold line.
- plot_lines(lines, x_vals, ax=None, logscale=True):
    Plots exponential decay lines (a, b) over specified x values.
- plot_period(problem, ax=None, **kwargs):
    Plots the phase portrait of the periodic solution from an HBMProblem.
Usage:
------
Run as a script to analyze two different Duffing oscillator parameter sets and visualize their solutions and spectral properties.
Dependencies:
-------------
- numpy
- matplotlib
- skhippr (systems.nonautonomous, problems.HBM, stability.KoopmanHillProjection, Fourier)
"""

import matplotlib.pyplot as plt
from copy import copy

from skhippr.systems.nonautonomous import duffing
from skhippr.problems.HBM import HBMEquation
from skhippr.stability.KoopmanHillProjection import KoopmanHillProjection

from skhippr.Fourier import Fourier


def main(params_duffing: dict[str, float], fourier: Fourier, x_init=None, idx_k=None):
    """
    Finds the periodic solution of the Duffing oscillator, identifies exponential decay parameters (a, b),
    and generates plots for the solution and its spectral properties.
    Parameters:
        params_duffing (dict[str, float]): Dictionary of Duffing oscillator parameters (keyword arguments to skhippr.systems.autonomous.duffing).
        fourier (Fourier): Discrete Fourier Transform configuration object.
        x_init (np.ndarray, optional): Initial guess for the state trajectory as a 2D array (shape: [2, N]). If None, a cosine/sine initial guess is used.
        idx_k (int or list[int], optional): Index or indices to select specific (a, b) lines from the computed exponential decay parameters.
    Returns:
        tuple:
            - lines (np.ndarray): Array representing the identified exponential decay parameters (a, b). lines[k, :] is k-th such tuple.
            - x_time (np.ndarray): Time series of the periodic solution.
    Side Effects:
        - Generates plots for the periodic solution and the spectral bar chart with identified lines.
        - Plots a comparison with the linearized system (beta=0) as a dashed black line.
    Raises:
        AssertionError: If the solver does not converge for either the nonlinear or linear problem.
    """

    # Initial guess for HBM problem
    if x_init is None:
        taus = fourier.time_samples(omega=1)
        x_init = np.vstack([np.cos(taus), -params_duffing["omega"] * np.sin(taus)])

    # Create and solve HBM problem
    prb = HBMEquation(
        f=duffing,
        fourier=fourier,
        stability_method=KoopmanHillProjection(fourier),
        tolerance=1e-8,
        verbose=True,
        period_k=1,
        parameters_f=params_duffing,
        omega=params_duffing["omega"],
        initial_guess=fourier.DFT(x_init),
    )
    prb.solve()
    assert prb.converged
    ax_period = plot_period(prb)

    # Identify exponential decay parameters
    threshold = 5e-15
    lines = prb.exponential_decay_parameters(threshold)

    if idx_k is not None:
        lines = lines[idx_k]
    try:
        lines[0][0]
    except IndexError:
        lines = [lines]

    # Bar chart with *all* norms and (a, b) lines
    fig, ax = plt.subplots(ncols=1, nrows=1)
    J_norms_all, ks_all = determine_J_norms(prb, threshold=0)
    plot_bar(ks_all, J_norms_all, threshold, ax)
    plot_lines(lines, ks_all, ax, logscale=True)
    ax.legend()

    # Solution of linear system (beta = 0) for reference
    params_linear = copy(params_duffing)
    params_linear["beta"] = 0
    prb_linear = prb = HBMEquation(
        f=duffing,
        fourier=fourier,
        stability_method=KoopmanHillProjection(fourier),
        tolerance=1e-8,
        verbose=True,
        period_k=1,
        parameters_f=params_linear,
        omega=params_linear["omega"],
        initial_guess=fourier.DFT(x_init),
    )
    prb_linear.solve()
    assert prb_linear.converged
    plot_period(prb_linear, ax=ax_period, linestyle="--", color="black")
    ax_period.legend()

    return lines, prb.x_time()


def determine_J_norms(prb: HBMEquation, threshold: float = 1e-15):
    """
    Computes the norms of the Fourier coefficients of the Jacobian in an HBMProblem.
    This function extracts the Fourier coefficient matrices from the provided
    HBMProblem instance, computes their 2-norms, and returns the norms
    and corresponding harmonic indices for those exceeding a specified threshold.
    Args:
        prb (HBMProblem): The harmonic balance method problem instance. Must be converged already to ensure that Fourier coefficients are available.
        threshold (float, optional): Minimum norm value to consider. Norms below this value are ignored.
            Defaults to 1e-15.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - norms: Array of 2-norms of the Jacobian matrices above the threshold.
            - ks: Array of corresponding harmonic indices for the returned norms.
    """
    Js = prb.ode_coeffs()
    norms = []
    ks = []
    for k in range(prb.fourier.N_HBM + 1):
        if prb.fourier.real_formulation:
            if k > 0:
                J_cplx = 0.5 * Js[:, :, k] + 0.5j * Js[:, :, prb.fourier.N_HBM + k]
            else:
                J_cplx = Js[:, :, 0]
        else:
            # consider the positive component
            J_cplx = Js[:, :, prb.fourier.N_HBM + k]

        norm_J = np.linalg.norm(J_cplx, ord=2)
        if norm_J > threshold:
            norms.append(norm_J)
            ks.append(k)
    return np.array(norms), np.array(ks)


def plot_bar(x_vals, y_vals, threshold: float = 0, ax=None):
    """
    Plots a logarithmic bar chart of the provided values with a horizontal threshold line.
    The x-axis is labeled as "$k$" and the y-axis as "$||J_k||$".

    Parameters
    ----------
    x_vals : array-like
        The x-coordinates of the bars (e.g., indices or categories).
    y_vals : array-like
        The heights of the bars.
    threshold : float, optional
        The y-value at which to draw a horizontal dashed line (default is 0).
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, a new figure and axes are created.
    Notes
    -----
    - The bar color is set to gray and the threshold line is black and dashed.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.bar(x_vals, y_vals, color="gray")
    plt.axhline(y=threshold, color="black", linestyle="dashed")
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$||J_k||$")

    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())


def plot_lines(
    lines: list[tuple[float, float]], x_vals: np.ndarray, ax=None, logscale=True
):
    """
    Plots a set of lines defined by (a, b) tuples over specified x values in a linar scale or logscale plot.

    Each line is defined as y(x) = a - b*x if logscale is False, or y(x) = a * exp(-b*x) if logscale is True.
    All lines are plotted on the provided matplotlib axis.

    Args:
        lines (list[tuple[float, float]]): List of (a, b) tuples defining the lines.
        x_vals (np.ndarray): Array of x values over which to plot the lines.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. If None, a new axis is created.
        logscale (bool, optional): If True, plot y(x) = a * exp(-b*x); otherwise, plot y(x) = a - b*x. Default is True.

    Returns:
        None
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    for k, (a, b) in enumerate(lines):
        y_vals = a - b * x_vals
        if logscale:
            y_vals = a * np.exp(-b * x_vals)
        else:
            y_vals = a - b * x_vals
        ax.plot(x_vals, y_vals, label=f"{k}: b = {b:.2f}, a={a:.2f}")


def plot_period(problem: HBMEquation, ax=None, **kwargs):
    """
    Plots the phase plot of a given HBMProblem instance.

    Parameters
    ----------
    problem : HBMProblem
        The problem instance containing the solution to be plotted. Must have a `x_time()` method
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, a new figure and axes will be created.
    **kwargs
        Additional keyword arguments passed to `ax.plot()`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted period response.

    Notes
    -----
    The plot will be labeled with the value of `problem.beta`.
    """
    x_time = problem.x_time()
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x_time[0, :], x_time[1, :], label=f"$\\beta={problem.beta}$", **kwargs)
    return ax


if __name__ == "__main__":
    params_config_1 = {"alpha": 5, "beta": 0.1, "delta": 0.02, "F": 0.1, "omega": 5}
    params_config_2 = {"alpha": 0.5, "beta": 3, "delta": 0.05, "F": 0.1, "omega": 0.3}

    fourier = Fourier(N_HBM=45, L_DFT=5 * 45, n_dof=2, real_formulation=True)
    _, x2_init = main(params_duffing=params_config_1, fourier=fourier, idx_k=0)
    main(params_duffing=params_config_2, fourier=fourier, x_init=x2_init, idx_k=2)
    plt.show()
