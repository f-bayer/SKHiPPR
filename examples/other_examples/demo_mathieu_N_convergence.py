"""Demonstration of convergence of Koopman-Hill projection method for the Mathieu equation.
This script analyzes the convergence of the Koopman-Hill projection method applied to the Mathieu equation,
a classical example of a parametrically excited system. The convergence is studied by increasing the number
of harmonics (N_HBM) in the Harmonic Balance Method (HBM) and comparing the resulting monodromy matrix to a
reference solution obtained via the shooting method.
Functions
---------
- analyze_N_mathieu(N_max=60):
    Runs the convergence analysis for the Mathieu equation up to N_max harmonics.
- analyze_N_hill(f, params, Phi_T_ref=None, N_max=10, params_plot=None, ax=None, csv_path=None, key_param=None):
    Analyzes the convergence of the Koopman-Hill projection for a general Hill-type problem by increasing
    the number of harmonics. Plots and optionally saves the error in the monodromy matrix.
- reference_monodromy_matrix(f, params, T):
    Computes the reference monodromy matrix for a given system using the shooting method, ensuring that
    the equilibrium at zero is maintained.
- setup_plot(ax, lambdas_ref):
    Sets up the plotting environment for visualizing convergence and Floquet multipliers.
- initialize_errors_with_param(params, key_param, csv_path):
    Initializes the error tracking list, optionally including a parameter value for CSV output.
- setup_hbm_problem(f, params, N_HBM):
    Configures and solves the HBM problem for the specified system and number of harmonics, ensuring
    convergence at the equilibrium.
- write_errors_to_csv(csv_path, errors):
    Appends the current error values to a CSV file.
Usage
-----
Run this script directly to perform the convergence analysis for the Mathieu equation and display the results.
Example:
    python demo_mathieu_N_convergence.py
Dependencies
------------
- numpy
- matplotlib
- csv
- skhippr (with modules: systems.ltp.mathieu, Fourier, problems.HBM, problems.shooting, stability.KoopmanHillProjection)
"""

from enum import auto
from colorama import init
import numpy as np
import matplotlib.pyplot as plt
import csv

from skhippr.systems.ltp import mathieu, meissner, meissner_g
from skhippr.Fourier import Fourier
from skhippr.problems.HBM import HBMProblem, HBMProblem_autonomous
from skhippr.problems.shooting import ShootingProblem
from skhippr.stability.KoopmanHillProjection import KoopmanHillProjection


def analyze_N_mathieu(N_max=60, csv_path=None):
    mathieu_params = {"a": 4, "b": 0.2, "omega": 1, "d": 0.005}
    if csv_path:
        initialize_csv(csv_path, N_max=N_max, key_param=None)
    analyze_N_convergence(mathieu, mathieu_params, N_max=N_max, csv_path=csv_path)


def analyze_smoothness_effects(N_max=30, csv_path=None, smoothing=None):
    """
    Analyze and visualize the effects of the smoothing parameter on the convergence of a Hill-type system.
    For smoothing==1, the system coincides with Mathieu's equation and for smoothing==0 with the Meissner equation.
    This function iterates over a range of smoothing parameter values, runs a convergence analysis for each,
    and plots both the convergence behavior and the time-domain response. Optionally, results can be saved to a CSV file.
    Parameters
    ----------
    N_max : int, optional
        Maximum number of harmonics or terms to consider in the convergence analysis (default is 30).
    csv_path : str or None, optional
        Path to a CSV file where results will be saved. If None, results are not saved (default is None).
    smoothing : array-like or None, optional
        Sequence of smoothing parameter values to analyze. If None, a default sequence from 1 to 0 is used.
    """
    params = {"a": 4, "b": 0.2, "omega": 1, "d": 0.005}
    T = 2 * np.pi / params["omega"]
    if smoothing is None:
        smoothing = np.flip(np.linspace(0, 1, 7, endpoint=True))

    if csv_path:
        initialize_csv(csv_path, N_max=N_max, key_param="smoothing")

    params_plot = dict()
    ax_conv, axs, _ = setup_plot(ax=None, lambdas_ref=0)
    ax_time = axs[1]
    ax_time.clear()
    ax_time.set_aspect("auto")
    ax_time.set_xlim((0, 0.5))
    ax_time.set_ylim((params["a"] - 1.2 * params["b"], params["a"] + 1.2 * params["b"]))
    ax_time.set_xlabel("t/T")

    for k, eps in enumerate(smoothing):
        params_plot["color"] = f"C{k}"
        params_plot["label"] = f"$\\epsilon = {eps}$"
        params["smoothing"] = eps
        hbm = analyze_N_convergence(
            meissner,
            params,
            N_max=N_max,
            csv_path=csv_path,
            params_plot=params_plot,
            ax_conv=ax_conv,
            key_param="smoothing",
            value_param=eps,
        )

        # Plot the considered function and the identified samples
        ts = hbm.fourier.time_samples(omega=params["omega"])

        g, _ = meissner_g(
            ts, params["a"], params["b"], params["omega"], params["smoothing"]
        )
        J_time = hbm.ode_samples()
        ax_time.plot(ts / T, g, "k-")
        ax_time.plot(ts / T, -J_time[1, 0, :].squeeze(), ".", **params_plot)

    ax_time.legend()
    ax_conv.legend()


def analyze_N_convergence(
    f,
    params,
    Phi_T_ref=None,
    N_max=10,
    params_plot=None,
    ax_conv=None,
    csv_path=None,
    key_param=None,
    value_param=None,
    variable="y",
    autonomous=False,
    initial_guess=None,
) -> HBMProblem:
    """
    Analyze the convergence of the Koopman-Hill projection with increasing number of harmonics (N_HBM)
    in the context of a Hill-type problem.
    Parameters
    ----------
    f : callable
        The function defining the system's equations. Must have an equilibrium at zero.
    params : dict
        Dictionary of parameters required by the system.
    Phi_T_ref : ndarray, optional
        Reference monodromy matrix for comparison. If None, it is computed using shooting.
    N_max : int, optional
        Maximum number of harmonics to consider (default is 10).
    params_plot : dict, optional
        Dictionary of plotting parameters (e.g., color, marker style).
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on. If None, a new axis is created.
    csv_path : str, optional
        Path to a CSV file where errors will be saved. If None, errors are not saved.
    key_param : str, optional
        Key of the parameter to track in the error output.
    Returns
    -------
    hbm: The HBM problem with N = N_max
    Side effects
    ------------
        The function plots the convergence of the monodromy matrix error and optionally saves errors to a CSV file.
    """

    # Setup
    T = 2 * np.pi / params["omega"]
    if autonomous:
        del params["omega"]

    if params_plot is None:
        params_plot = {"color": "k"}

    if initial_guess is None:
        initial_guess = np.array([0.0, 0.0])

    if Phi_T_ref is None:
        Phi_T_ref, sol_shoot = reference_monodromy_matrix(
            f,
            params,
            T,
            variable=variable,
            autonomous=autonomous,
            initial_guess=initial_guess,
        )
    else:
        sol_shoot = None
    lambdas_ref = np.linalg.eig(Phi_T_ref).eigenvalues

    ax_conv, axs, plot_FMs = setup_plot(ax_conv, lambdas_ref)

    errors = initialize_errors_with_param(params, key_param, csv_path)

    for N_HBM in range(1, N_max + 1):
        print(N_HBM)
        hbm = setup_hbm_problem(
            f,
            params,
            N_HBM,
            key_param,
            value_param,
            autonomous=autonomous,
            sol_ref=sol_shoot,
            variable=variable,
        )

        lambdas = hbm.eigenvalues
        Phi_T = hbm.stability_method.fundamental_matrix(t_over_period=1, problem=hbm)

        errors.append(np.linalg.norm(Phi_T - Phi_T_ref, ord=2))
        ax_conv.semilogy(N_HBM, errors[-1], ".", **params_plot)
        try:
            del params_plot["label"]
        except:
            # The key doesn't exist
            pass

        if plot_FMs:
            axs[1].plot(np.real(lambdas), np.imag(lambdas), ".")
        pass

    if csv_path:
        write_errors_to_csv(csv_path, errors)

    return hbm


def reference_monodromy_matrix(
    f, params, T, variable="y", initial_guess=None, autonomous=False, L_DFT=1026
):
    """
    Computes the reference monodromy matrix for a given system with an equilibrium at zero using the shooting method.

    This function sets up and solves the shooting problem for a parametrically excited system of ODEs defined by `f`
    with the specified parameters and period `T`.
    It asserts that 0 is indeed a solution. The function returns the monodromy matrix.

    Args:
        f (callable): The system of ODEs to solve. Should accept state, time, and parameters.
        params (dict): Parameters to pass to the ODE system.
        T (float): The period over which to solve the shooting problem.
        variable (str): state variable of f.
        initial_guess(np.ndarray): Initial guess for point on periodic solution.
                                   If not passed, the function assumes that there exists an equilibrium at 0.
        autonomous(bool): Whether the considered system is autonomous.

    Returns:
        numpy.ndarray: The reference fundamental matrix (2x2 array).
        sol_shoot (ShootingProblem): The whole shooting problem

    Raises:
        AssertionError: If 0 is not an equilibrium and no initial guess was passed.
    """

    if initial_guess is None:
        x0 = np.array([0.0, 0.0])
    else:
        x0 = initial_guess

    params_odesolver = {"atol": 1e-12, "rtol": 1e-12}
    sol_shooting = ShootingProblem(
        f=f,
        x0=x0,
        T=T,
        autonomous=autonomous,
        variable=variable,
        verbose=True,
        parameters=params,
        kwargs_odesolver=params_odesolver,
    )
    sol_shooting.solve()
    assert sol_shooting.converged
    if initial_guess is None:
        # solved at equilibrium point (0,0)
        assert np.max(np.abs(sol_shooting.x)) == 0

    x_time = sol_shooting.x_time(
        t_eval=np.linspace(0, 2 * np.pi / sol_shooting.omega, L_DFT, endpoint=False)
    )
    return sol_shooting.derivatives[variable][:2, :2] + np.eye(2), sol_shooting


def setup_plot(ax, lambdas_ref):
    """
    Sets up a matplotlib plot for visualizing reference eigenvalues and optionally Floquet multipliers.
    If ax is None, a new figure with two subplots is created. Plot_FMs is set to True.
    Otherwise, ax is simply returned and plot_FMs is set to False.

    Parameters:
        ax (matplotlib.axes.Axes or None): The axis to plot on. If None, a new figure with two subplots is created.
        lambdas_ref (array-like): Reference eigenvalues to be plotted in the complex plane.

    Returns:
        tuple:
            - ax (matplotlib.axes.Axes): The axis to be used for further plotting.
            - axs (array-like or None): The array of axes if new subplots were created, otherwise None.
            - plot_FMs (bool): Flag indicating whether Floquet multipliers should be plotted.
    """
    if ax is None:
        _, axs = plt.subplots(ncols=2, nrows=1)
        phis = np.linspace(0, 2 * np.pi, 250)
        axs[1].plot(np.cos(phis), np.sin(phis), "--", color="gray")
        axs[1].plot(np.real(lambdas_ref), np.imag(lambdas_ref), "kx")
        axs[1].axis("equal")
        axs[1].set_xlabel("Re(\\lambda)")
        axs[1].set_ylabel("Im(\\lambda)")
        del phis
        plot_FMs = True
        ax = axs[0]
        ax.set_ylabel("E")
        ax.set_xlabel("N")
    else:
        axs = None
        plot_FMs = False
    return ax, axs, plot_FMs


def initialize_errors_with_param(params, key_param, csv_path):
    """
    Initializes a list of errors.
    If a CSV path is provided, the first column of the CSV corresponds to different parameter values. Then, the list is initialized with that value.
    Otherwise, the list is initialized empty.

    Args:
        params (dict): Dictionary containing configuration parameters. May include the key "smoothing".
        csv_path (str or None): Path to a CSV file. If provided (not None or empty), the smoothing parameter is used.

    Returns:
        list: A list containing the smoothing value from params if csv_path is provided; otherwise, an empty list.
    """
    if csv_path:
        return [params.get(key_param, 0)]  # fallback if smoothing not present
    else:
        return []


def setup_hbm_problem(
    f,
    params_default,
    N_HBM,
    key_param=None,
    value_param=None,
    autonomous=False,
    sol_ref=None,
    variable="y",
):
    """
    Sets up and solves a Harmonic Balance Method (HBM) problem for a given function and parameters. The function must have an equilibrium at 0.

    Parameters:
        f (callable): The function representing the system to be analyzed. Must have an equilibrium at 0.
        params (dict): Dictionary containing problem parameters, must include "omega".
        N_HBM (int): Number of harmonics to use in the HBM.

    Returns:
        HBMProblem: The solved HBMProblem instance.

    Raises:
        AssertionError: If the HBM problem does not converge or the solution is not 0
    """
    fourier_kwargs = {"L_DFT": 1026, "real_formulation": True}
    if key_param in fourier_kwargs:
        fourier_kwargs[key_param] = value_param

    fourier = Fourier(N_HBM=N_HBM, n_dof=2, **fourier_kwargs)
    if sol_ref:
        X_init = fourier.DFT(sol_ref.x_time(fourier.time_samples(sol_ref.omega)))
        omega_init = sol_ref.omega
    else:
        X_init = np.zeros((2 * fourier.N_HBM + 1) * fourier.n_dof)
        omega_init = 1
    if autonomous:
        hbm = HBMProblem_autonomous(
            f=f,
            omega=omega_init,
            initial_guess=X_init,
            variable=variable,
            stability_method=KoopmanHillProjection(fourier),
            verbose=False,
            fourier=fourier,
            parameters_f=params_default,
        )
    else:
        hbm = HBMProblem(
            f=f,
            omega=params_default["omega"],
            initial_guess=X_init,
            variable=variable,
            stability_method=KoopmanHillProjection(fourier),
            verbose=False,
            fourier=fourier,
            parameters_f=params_default,
        )
    if key_param is not None and key_param not in fourier_kwargs:
        hbm.__setattr__(key_param, value_param)
    hbm.solve()
    assert hbm.converged
    if sol_ref is None:
        assert np.max(np.abs(hbm.x)) == 0
    return hbm


def write_errors_to_csv(csv_path, errors):
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(errors)


def initialize_csv(csv_path, N_max, key_param=None):
    header = list(range(N_max))
    if key_param is not None:
        header = [key_param] + header

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(header)


if __name__ == "__main__":
    analyze_N_mathieu(N_max=10, csv_path="data_mathieu.csv")
    vals_smoothing = np.insert(np.logspace(-3, 0, 4, endpoint=True), 0, 0)
    analyze_smoothness_effects(
        N_max=70,
        csv_path="data_meissner_smoothed.csv",
        smoothing=vals_smoothing,
    )
    plt.show()
