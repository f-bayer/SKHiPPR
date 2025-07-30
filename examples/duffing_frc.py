"""
Analyze oscillations of the Duffing oscillator using SKHiPPR by performing
continuation along both excitation frequency and excitation amplitude.

"""

from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # for the progress bar

# --- Fourier configuration ---
from skhippr.Fourier import Fourier

# --- System function ---
from skhippr.equations.odes.nonautonomous import duffing

# --- HBM solver ---
from skhippr.problems.HBM import HBMEquation

# --- Stability method ---
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator, BranchPoint


def main():
    """
    Runs a demonstration for analyzing the Duffing oscillator with HBM, to illustrate SKHiPPR usage.

    This function performs the following steps:

    #. Parameter setup.
    #. Creation of a :py:class:`~skhippr.Fourier` object to collect FFT parameters.
    #. Creation of a :py:class:`~skhippr.stability.KoopmanHillProjection.KoopmanHillProjection` object which defines the stability method. Other :py:class:`~skhippr.stability._StabilityHBM` subclasses can interchangeably be used, such as:

    * :py:class:`~skhippr.stability.KoopmanHillProjection.KoopmanHillSubharmonic`
    * :py:class:`~skhippr.stability.ClassicalHill.ClassicalHill`
    * :py:class:`~skhippr.stability.SinglePass.SinglePassRK`

    #. Setup and solution of the initial :py:class:`~skhippr.problems.HBM.HBMProblem`.
    #. Visualization of the initial solution and its stability properties using :py:func:`visualize_solution`.
    #. Continuation of the force response curve using :py:func:`initial_force_response`
    #. Generation of frequency response curves from starting from points on the previous force response using :py:func:`continue_from_continuation_curve`.
    #. Generation of additional force responses starting from points on the frequency response curve using :py:func:`continue_from_continuation_curve`.
    #. 3D plotting of all computed response curves, including stability information, using :py:func:`plot_3D_frc`.

    Returns
    -------
    None
    """
    # --- Parameter setup ---
    omega_min, omega_max = 0.8, 12
    omegas_spark = [0.3, 0.8, 1.1, 1.7, 4]
    F_min, F_max = 0.05, 20
    Fs_spark = [1, 5, 10, 17]

    # --- FFT and stability method configuration ---
    fourier = Fourier(N_HBM=25, L_DFT=300, n_dof=2, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(
        fourier=fourier, tol=1e-4, autonomous=False
    )

    # --- Initial guess in time and frequency domain ---
    ts = fourier.time_samples(omega_min)
    x0_samples = np.array([np.cos(ts * omega_min), -omega_min * np.sin(ts * omega_min)])
    X0 = fourier.DFT(x0_samples)

    # --- HBM problem setup ---
    parameters = {
        "omega": omega_min,
        "F": F_min,
        "alpha": 1,
        "beta": 0.2,
        "delta": 0.1,
    }
    initial_problem = HBMEquation(
        f=duffing,
        omega=omega_min,
        initial_guess=X0,
        fourier=fourier,
        stability_method=stability_method,
        period_k=1,
        parameters_f=parameters,
    )

    # --- Solve initial HBM problem and visualize ---
    initial_problem.solve()
    print(initial_problem)
    assert initial_problem.converged
    visualize_solution(initial_problem)

    # --- Force response continuation ---
    response_F = initial_force_response(initial_problem, F_min, F_max, verbose=True)

    # --- Frequency response curves from force response points ---
    responses_omega = continue_from_continuation_curve(
        curve=response_F,
        key_cont="omega",
        param_range=(omega_min, omega_max),
        values_spark=Fs_spark,
    )

    # --- Additional force responses from frequency response points ---
    responses_F = continue_from_continuation_curve(
        responses_omega[0],
        key_cont="F",
        param_range=(F_min, F_max),
        values_spark=omegas_spark,
    )

    # --- Plot results ---
    plot_all_responses(response_F, responses_omega + responses_F)


def initial_force_response(
    initial_problem: HBMEquation, F_min: float, F_max: float, verbose=True
) -> list[BranchPoint]:
    """
    Continue the force response curve starting from the initial solution.

    This function uses the :py:func:`~skhippr.problems.continuation.pseudo_arclength_continuator` to trace the response of a system as the force parameter ``F`` is varied from ``F_min`` up to ``F_max``.

    Args
    ----
    initial_problem : :py:class:`~skhippr.problems.HBM.HBMProblem`
        The initial problem or problem setup to start the continuation from. Must take a parameter ``F``.
    F_min : float
        The minimum value of the force parameter ``F`` to start the continuation.
    F_max : float
        The maximum value of the force parameter ``F`` at which to stop the continuation.
    verbose : bool, optional
        If ``True``, enables verbose output during continuation. Defaults to ``True``.

    Returns
    -------
    response_F : list[BranchPoint]
        A list of :py:class:`~skhippr.problems.continuation.BranchPoint`` objects representing the computed points along the force response curve.
    """

    response_F = []
    for branch_point_F in pseudo_arclength_continuator(
        initial_problem=initial_problem,
        stepsize=0.1,
        stepsize_range=(0.01, 0.5),
        initial_direction=1,
        key_param="F",
        value_param=F_min,
        verbose=verbose,
    ):
        response_F.append(branch_point_F)
        if branch_point_F.F > F_max:
            break
    return response_F


def continue_from_continuation_curve(
    curve: list[BranchPoint],
    key_cont: str,
    param_range: tuple[float, float],
    values_spark: list[float],
) -> list[list[BranchPoint]]:
    """
    Finds and computes frequency response curves emanating from specified points on a force response curve.

    This function iterates over a given continuation curve (list of :py:class:`~skhippr.problems.continuation.BranchPoint` objects). When the continuation parameter first exceeds the next value in ``values_spark``, the method it initiates a new continuation along the different parameter ``key_cont`` (e.g., ``"omega"``). The resulting branches are collected and returned as a list of lists of :py:class:`~skhippr.problems.continuation.BranchPoint` objects.

    Parameters
    ----------
    curve : list[BranchPoint]
        The initial continuation curve.
    key_cont : str
        The name of the new continuation parameter (e.g., `'"omega"'`).
    param_range : tuple[float, float]
        The minimum and maximum values for the (new) continuation parameter.
    values_spark : list[float]
        The values of the (old) continuation parameter of the initial curve at which to start new continuation curves along the new parameter.

    Returns
    -------
    responses: list[list[BranchPoint]]
        A list containing the new continuation curves.
    """
    responses = []
    param_min, param_max = param_range
    for branch_point_init in curve:
        if not values_spark:
            break
        if branch_point_init.x[-1] > values_spark[0]:
            values_spark.pop(0)
            responses.append([])

            pbf = f"{key_cont} curve at {branch_point_init.key_param} = {branch_point_init.x[-1]:5.2f}: {key_cont} = {{n:5.2f}} |{{bar}}| "
            with tqdm(total=param_max, bar_format=pbf) as progress_bar:
                for branch_point in pseudo_arclength_continuator(
                    initial_problem=branch_point_init.copy_problem(),
                    stepsize=0.1,
                    stepsize_range=(0.01, 0.25),
                    initial_direction=1,
                    key_param=key_cont,
                    value_param=getattr(branch_point_init, key_cont),
                    verbose=False,
                    num_steps=10000,
                ):
                    progress_bar.n = getattr(branch_point, key_cont)
                    progress_bar.refresh()
                    responses[-1].append(branch_point)
                    if not (param_min <= getattr(branch_point, key_cont) <= param_max):
                        break
    return responses


def plot_all_responses(
    initial_response: list[BranchPoint], other_responses: list[list[BranchPoint]]
):
    """
    Plot all computed response curves using py:func:`plot_3D_frc`.

    Parameters
    ----------
    initial_response : list[BranchPoint]
        The initial response curve to be plotted.
    other_responses : list[list[BranchPoint]]
        A list of additional response curves to plot.

    """
    ax = plot_3D_frc(initial_response, "Initial response", plot_stability=True)
    for response in other_responses:
        plot_3D_frc(response, ax=ax, plot_stability=True)


def plot_3D_frc(
    list_of_points: Iterable[BranchPoint], label="", ax=None, plot_stability=True
):
    """
    Plot a 3D curve of branch points with stability information.

    Visualize a list of BranchPoint objects in 3D space, where the axes represent
    the frequency (``omega``), forcing amplitude (``F``), and the maximum absolute value of the first
    state variable (``|x_1|``). Points can be colored according to their stability.

    Parameters
    ----------
    list_of_points : Iterable[BranchPoint]
        A continuation curve to plot. Every :py:class:`~skhippr.problems.continuation.BranchPoint` must have the attributes ``point.F`` and ``point.omega``.
    label : str, optional
        Label for the curve. Defaults to ``""``.
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        Existing 3D axes to plot on. If ``None``, a new figure and axes are created. Defaults to ``None``.
    plot_stability : bool, optional
        Whether to highlight stable and unstable points with different colors. Defaults to ``True``.

    Returns
    -------
    ax: matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes with the plotted data.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    stable = np.array([point.stable for point in list_of_points])
    omegas = np.array([point.omega for point in list_of_points])
    Fs = np.array([point.F for point in list_of_points])
    amplitudes = np.array(
        [np.max(np.abs(point.x_time()[0, :])) for point in list_of_points]
    )

    ax.plot(omegas, Fs, amplitudes, label=label)
    if plot_stability:
        ax.plot(
            omegas[stable],
            Fs[stable],
            amplitudes[stable],
            "r.",
            label="stable",
            markersize=1,
        )
        ax.plot(
            omegas[~stable],
            Fs[~stable],
            amplitudes[~stable],
            "b.",
            label="unstable",
            markersize=1,
        )

    ax.set_xlabel("omega")
    ax.set_ylabel("F")
    ax.set_zlabel("|x_1|")
    return ax


def visualize_solution(problem: HBMEquation):
    """
    Visualizes and analyzes properties of one solved :py:class:`~skhippr.problems.HBM.HBMProblem`.

    This function generates two subplots:

    #. A phase plot of the solution, plotting the first two state variables over time.
    #. A plot of the Floquet multipliers in the complex plane, along with the unit circle for reference.

    Parameters
    ----------
    problem : HBMProblem
        The :py:class:`~skhippr.problems.HBM.HBMProblem` object containing the state trajectory and Floquet multipliers. It is assumed that ``problem.converged == True``.

    Returns
    -------
        None
    """
    _, axs = plt.subplots(nrows=1, ncols=2)
    x_time = problem.x_time()
    axs[0].plot(x_time[0, :], x_time[1, :])
    axs[0].set_title("Phase plot of solution")
    axs[0].set_ylabel("x_1")
    axs[0].set_xlabel("x_0")

    floquet_multipliers = problem.eigenvalues
    axs[1].plot(np.real(floquet_multipliers), np.imag(floquet_multipliers), "x")
    axs[1].set_title("Floquet multipliers")
    axs[1].plot(
        np.cos(problem.fourier.time_samples_normalized),
        np.sin(problem.fourier.time_samples_normalized),
        "k",
    )
    axs[1].axis("equal")


if __name__ == "__main__":
    main()
    plt.show()
