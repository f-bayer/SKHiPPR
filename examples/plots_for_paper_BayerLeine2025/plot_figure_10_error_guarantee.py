"""
This script analyzes and visualizes the error guarantees for the Koopman-Hill projection method within the Harmonic Balance Method (HBM) framework, specifically for the Duffing oscillator system.
This recreates Figure 10 from Bayer&Leine2025.

Main Functionalities:
---------------------
- Computes the reference fundamental matrix solution using the shooting method.
- Evaluates both numerical error and theoretical error bound for the fundamental solution matrix using Koopman-Hill projection.
- Determines the minimum number of harmonics required to achieve a desired error threshold, both numerically and theoretically.
- Provides plotting utilities to visualize:
    - Error evolution over time for a fixed HBM order.
    - Error as a function of HBM order for a fixed time.
    - Required number of harmonics over time to achieve a specified error tolerance.

Functions:
--------------
- Phi_t_ref: Computes the reference fundamental matrix and state trajectory using shooting.
- errors_koopman_hill: Calculates numerical and theoretical error bounds for the fundamental solution matrix.
- N_koopman_hill: Determines the minimum number of harmonics needed to meet a desired error threshold.
- plot_error_over_t: Plots error evolution over time for a given HBM order.
- plot_error_over_N: Plots error as a function of HBM order for a fixed time.
- plot_N_over_t: Plots the required number of harmonics over time for a specified error tolerance.

Usage:
Run the script to generate and display the error analysis plots for the Duffing oscillator using the Koopman-Hill projection method.
Dependencies:
-------------
- numpy
- matplotlib
- skhippr (Fourier, systems.nonautonomous, problems.shooting, problems.hbm, stability.KoopmanHillProjection)
"""

import numpy as np
import matplotlib.pyplot as plt

from pytest import param
from skhippr.Fourier import Fourier
from skhippr.odes.nonautonomous import duffing
from skhippr.problems.shooting import ShootingProblem
from skhippr.problems.hbm import HBMEquation
from skhippr.stability.KoopmanHillProjection import KoopmanHillProjection


def Phi_t_ref(
    ts, problem_ref: HBMEquation, ode_kwargs: dict[str, float] = None
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Computes the reference fundamental matrix solution using shooting at a given set of time samples.
    Also returns the state trajectory.
    Parameters
    ----------
    ts : array-like
        Array of time samples at which to evaluate the fundamental matrix.
    problem_ref : HBMProblem
        Reference harmonic balance method problem containing system parameters and frequency.
    ode_kwargs : dict[str, float], optional
        Dictionary of keyword arguments to pass to the ODE solver (e.g., 'atol', 'rtol'). Defaults to {'atol': 1e-8, 'rtol': 1e-8}.
    Returns
    -------
    x_time : np.ndarray
        State trajectory evaluated at the linearly spaced time samples required for DFT.
    Phi_ts : list[np.ndarray]
        List of fundamental matrix solutions evaluated at the time samples `ts`.
    Raises
    ------
    AssertionError
        If the shooting problem does not converge.
    """

    if ode_kwargs is None:
        ode_kwargs = {"atol": 1e-13, "rtol": 1e-13}

    T = 2 * np.pi / problem_ref.omega

    problem_shooting = ShootingProblem(
        f=duffing,
        x0=np.array([0.2, 0.0]),
        T=T,
        tolerance=1e-12,
        verbose=True,
        kwargs_odesolver=ode_kwargs,
        parameters=problem_ref.get_params(),
    )
    problem_shooting.solve()
    assert problem_shooting.converged

    # Temporarily set ts as a keyword arg to the ode solver
    problem_shooting.kwargs_odesolver["t_eval"] = ts
    _, _, Phi_ts = problem_shooting.integrate_with_fundamental_matrix(
        problem_shooting.x,
        T,
    )

    # Remove t_eval as kwarg because it is explicitly set in x_time
    del problem_shooting.kwargs_odesolver["t_eval"]
    x_time = problem_shooting.x_time(
        t_eval=problem_ref.fourier.time_samples(params["omega"])
    )

    return x_time, Phi_ts


def errors_koopman_hill(
    ts, N_HBM, problem_ref: HBMEquation, Phi_ref, params_decay=None
):
    """
    Computes the numerical and theoretical error bounds for the fundamental solution matrix
    of a system solved using the Koopman-Hill projection method within the Harmonic Balance Method (HBM) framework.
    Parameters
    ----------
    ts : array-like
        Array of time points at which to evaluate the fundamental solution matrix and errors.
    N_HBM : int
        Number of harmonics to use in the Harmonic Balance Method.
    problem_ref : HBMProblem
        Reference HBMProblem instance containing the system definition and reference solution. Used to compute exponential decay if not passed.
    Phi_ref : ndarray
        Reference fundamental solution matrix of shape (n, n, len(ts)), where n is the system dimension.
    params_decay: ndarray, optional
        Exponential decay parameters as a result of problem_ref.
    Returns
    -------
    errors_num : ndarray
        Array of numerical errors (2-norm) between the computed and reference fundamental solution matrices at each time in `ts`.
    errors_bound : ndarray
        Array of theoretical error bounds for the fundamental solution matrix at each time in `ts`.
    """

    # Set up problem
    fourier = problem_ref.fourier.__replace__(N_HBM=N_HBM)
    params = problem_ref.get_params()
    problem = HBMEquation(
        f=duffing,
        initial_guess=fourier.DFT(problem_ref.x_time()),
        fourier=fourier,
        omega=params["omega"],
        stability_method=KoopmanHillProjection(fourier),
        parameters_f=params,
        verbose=False,
    )
    problem.solve()
    assert problem.converged

    # Determine fundamental solution matrix and error
    T = 2 * np.pi / params["omega"]
    errors_num = np.zeros_like(ts)

    for k, t in enumerate(ts):
        Phi_t = problem.stability_method.fundamental_matrix(
            t_over_period=t / T, problem=problem
        )
        errors_num[k] = np.linalg.norm(Phi_t - Phi_ref[:, :, k], ord=2)

    # Error bound
    if params_decay is None:
        params_decay = problem_ref.exponential_decay_parameters()
    errors_bound = problem.error_bound_fundamental_matrix(
        ts, params_decay[:, 0], params_decay[:, 1]
    )

    return errors_num, errors_bound


def N_koopman_hill(ts, N_max, E_des, problem_ref, Phi_ref, params_decay):
    """
    Computes the minimum number of harmonics required to guarantee a desired error threshold
    for a given set of time points, using both a theoretical error bound and a numerical check.
    Parameters
    ----------
    ts : array_like
        Array of time points at which the error is evaluated.
    N_max : int
        Maximum number of harmonics to consider in the numerical search.
    E_des : float
        Desired error threshold for the Koopman-Hill approximation.
    problem_ref : object
        Reference to the problem instance, expected to be compatible with `errors_koopman_hill`.
    Phi_ref : object
        Reference fundamental solution, passed to `errors_koopman_hill`.
    params_decay : list or None
        List of tuples (a, b) representing exponential decay parameters. If None, these are obtained
        from `problem_ref.exponential_decay_parameters()`.
    Returns
    -------
    N_num : ndarray
        Array of the minimum number of harmonics required at each time point in `ts` to achieve
        the desired error threshold, determined numerically.
    N_bound : ndarray
        Array of theoretical upper bounds on the number of harmonics required at each time point
        in `ts` to guarantee the desired error threshold, computed from the decay parameters.
    """
    # decay parameters
    if params_decay is None:
        params_decay = problem_ref.exponential_decay_parameters()

    # error bound
    N_bound = np.inf * np.ones_like(ts)
    for a, b in params_decay:
        N_next = (np.abs(4 * a * ts) - np.log(E_des)) / (b - np.log(2))
        N_bound = np.minimum(N_bound, N_next)

    N_num = np.nan * np.ones_like(ts)

    for N in range(N_max):

        errors_num, _ = errors_koopman_hill(ts, N, problem_ref, Phi_ref, params_decay)
        mask_replace = np.isnan(N_num) & (errors_num <= E_des)
        N_num[mask_replace] = N

        if not np.any(np.isnan(N_num)):
            break
    return N_num, N_bound


def plot_error_over_t(ts, N_HBM, problem_ref, Phi_ref, params_decay):
    """
    Plots the numerical and guaranteed error over time for a given problem and HBM order.

    Parameters:
        ts (np.ndarray): Array of time points at which to evaluate the errors.
        N_HBM (int): Order of the Harmonic Balance Method (HBM).
        problem_ref: Reference problem object containing system parameters, including 'omega'.
        Phi_ref: Reference solution or basis used for error computation.
        params_decay: Parameters controlling the decay or error estimation.

    Returns:
        None. Displays a matplotlib figure with the error curves.
    """
    T = 2 * np.pi / problem_ref.omega
    e_num, e_guar = errors_koopman_hill(ts, N_HBM, problem_ref, Phi_ref, params_decay)
    fig, ax = plt.subplots(1, 1)
    ax.plot(ts / T, e_guar, label="error bound")
    ax.plot(ts / T, e_num, "--", label="actual error")
    ax.set_yscale("log")
    ax.set_xlabel("t/T")
    ax.set_ylabel("E")
    ax.set_title(f"N={N_HBM}")
    ax.legend()


def plot_error_over_N(t, Ns_HBM, problem_ref, Phi_ref, params_decay):
    """
    Plots the numerical and guaranteed error over different values of N for a given time.
    Parameters:
        t (float): The time at which the error is evaluated.
        Ns_HBM (array-like): Array of integer values representing the different N values to evaluate.
        problem_ref (object): Reference HBMproblem.
        Phi_ref (np.ndarray): Reference fundamental solution matrix, expected to be 2D (will be expanded to 3D).
        params_decay (object): Parameters related to the decay, passed to the error computation function.
    Returns:
        None. Displays a matplotlib figure showing the error bound and actual error as a function of N.
    """
    T = 2 * np.pi / problem_ref.omega
    Phi_ref = Phi_ref[:, :, np.newaxis]
    es_num = []
    es_guar = []
    for N in Ns_HBM:
        e_num, e_guar = errors_koopman_hill([t], N, problem_ref, Phi_ref, params_decay)
        es_num.append(e_num[0])
        es_guar.append(e_guar[0])
    es_num = np.array(es_num)
    es_guar = np.array(es_guar)

    fig, ax = plt.subplots(1, 1)
    ax.plot(Ns_HBM, es_guar, label=f"error bound")
    ax.plot(Ns_HBM, es_num, "--", label="actual error")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("E")
    ax.set_title(f"t/T = {t/T}")
    ax.legend()


def plot_N_over_t(ts, E_des, problem_ref, Phi_ref, params_decay):
    """
    Plots the evolution of the required number of basis functions N over time for a given desired error E_des.
    This function computes both the guaranteed upper bound and the actual number of harmonics needed to achieve the desired error tolerance, and visualizes them on a logarithmic scale.
    Parameters:
        ts (array-like): Array of time points at which to evaluate N.
        E_des (float): Desired error tolerance.
        problem_ref: reference HBM problem.
        Phi_ref: Reference fundamental solution matrix.
        params_decay: Parameters controlling the decay properties for the computation.
    Returns:
        None. Displays a matplotlib figure with the plot.
    """

    N_num, N_bound = N_koopman_hill(ts, 60, E_des, problem_ref, Phi_ref, params_decay)

    fig, ax = plt.subplots(1, 1)
    ax.plot(ts, N_bound, label=f"N* (guaranteed)")
    ax.plot(ts, N_num, "--", label="N*(actual)")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("N")
    ax.set_title(f"E_{{des}} = {E_des}")
    ax.legend()


if __name__ == "__main__":
    params = {"alpha": 0.5, "beta": 3, "delta": 0.05, "F": 0.1, "omega": 0.3}
    fourier_ref = Fourier(N_HBM=40, L_DFT=512, n_dof=2, real_formulation=True)
    problem_ref = HBMEquation(
        duffing,
        np.zeros(fourier_ref.n_dof * (2 * fourier_ref.N_HBM + 1)),
        params["omega"],
        fourier_ref,
        variable="x",
        stability_method=KoopmanHillProjection(fourier_ref),
        parameters_f=params,
    )
    ts = np.linspace(0, 2 * np.pi / params["omega"], 100, endpoint=True)

    x_init, Phi_ref = Phi_t_ref(ts, problem_ref)

    problem_ref.reset(x0_new=problem_ref.fourier.DFT(x_init))
    problem_ref.solve()
    assert problem_ref.converged
    params_decay = problem_ref.exponential_decay_parameters()

    plot_error_over_t(ts, 8, problem_ref, Phi_ref, params_decay)
    plot_error_over_N(
        t=2 * np.pi / params["omega"],
        Ns_HBM=np.arange(1, fourier_ref.N_HBM + 1),
        problem_ref=problem_ref,
        Phi_ref=Phi_ref[:, :, -1],
        params_decay=params_decay,
    )
    plot_N_over_t(ts, 1e-5, problem_ref, Phi_ref, params_decay)
    plt.show()
