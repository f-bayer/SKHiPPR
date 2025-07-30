"""Test HBM using the Duffing oscillator. See also: Matlab workshop, Task x.x"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import copy
from matplotlib.animation import FuncAnimation
import pytest

from skhippr.Fourier import Fourier
from skhippr.problems.HBM import HBMSystem
from skhippr.problems.shooting import ShootingSystem
from skhippr.systems.nonautonomous import Duffing
from skhippr.systems.autonomous import Vanderpol
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic
from skhippr.problems.continuation import pseudo_arclength_continuator
from skhippr.problems.newton import NewtonSolver


@pytest.fixture(params=[True, False])
def ode(request):
    autonomous = request.param
    if autonomous:
        return Vanderpol(t=0, x=np.array([2.0, 0]), nu=0.05)
    else:
        return Duffing(
            t=0,
            x=np.array([1.0, 0.0]),
            omega=0.8,
            alpha=1,
            beta=0.2,
            delta=0.1,
            F=3,
        )


@pytest.fixture
def shooting_system(ode):
    if ode.autonomous:
        omega = 1
    else:
        omega = ode.omega
    return ShootingSystem(ode=copy(ode), T=2 * np.pi / omega, atol=1e-4, rtol=1e-5)


@pytest.fixture
def hbm_system(ode):

    if ode.autonomous:
        omega = 1
    else:
        omega = ode.omega

    fourier = Fourier(N_HBM=25, L_DFT=300, n_dof=ode.n_dof, real_formulation=True)

    ts = fourier.time_samples(omega)
    x0_samples = np.vstack((np.cos(omega * ts), -omega * np.sin(omega * ts)))
    X0 = fourier.DFT(x0_samples)

    stability_method = KoopmanHillSubharmonic(fourier=fourier)

    return HBMSystem(
        ode=copy(ode),
        initial_guess=X0,
        omega=omega,
        fourier=fourier,
        stability_method=stability_method,
    )


def run_continuation(system, solver, parameter, param_range, max_steps=200):
    branch = []

    for bp in pseudo_arclength_continuator(
        initial_system=system,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.001, 0.4),
        initial_direction=1,
        num_steps=max_steps,
        continuation_parameter=parameter,
        verbose=True,
    ):
        branch.append(bp)

        if not param_range[0] <= getattr(bp, parameter) <= param_range[1]:
            break
    return branch


def bif_curve_nonaut(real_formulation=False):

    class Struct:
        pass

    request = Struct()
    request.autonomous = False

    ode = ode(request)
    omega = ode.omega

    N_HBM = 25
    L_DFT = 300

    fourier = Fourier(
        N_HBM=N_HBM, L_DFT=L_DFT, n_dof=2, real_formulation=real_formulation
    )

    print("Duffing oscillator")

    ts = fourier.time_samples(omega)
    x0_samples = np.vstack((np.cos(omega * ts), -omega * np.sin(omega * ts)))
    X0 = fourier.DFT(x0_samples)

    stability_method = KoopmanHillSubharmonic(fourier=fourier)

    hbm_system = HBMSystem(
        ode=ode,
        initial_guess=X0,
        omega=omega,
        fourier=fourier,
        stability_method=stability_method,
    )

    shooting_system = ShootingSystem(
        ode=copy(ode), T=2 * np.pi / ode.omega, atol=1e-5, rtol=1e-5
    )

    F_range = (0, 20)

    fig_3d = plt.figure()
    ax = fig_3d.add_subplot(111, projection="3d")
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_zlabel("F")

    fig_stability = plt.figure()
    solver = NewtonSolver()

    for initial_system in (hbm_system, shooting_system):

        xs_time = []
        amplitudes = []
        stable = []
        Fs = []

        for branch_point in pseudo_arclength_continuator(
            initial_system=initial_system,
            solver=solver,
            stepsize=0.1,
            stepsize_range=(0.001, 0.15),
            initial_direction=1,
            num_steps=1000,
            continuation_parameter="F",
            verbose=True,
        ):
            xs_time.append(branch_point.equations[0].x_time())
            amplitudes.append(np.max(xs_time[-1][0, :]))
            Fs.append(branch_point.F)
            stable.append(branch_point.stable)

            branch_point.determine_tangent()
            x_tng = branch_point.vector_of_unknowns + 0.2 * branch_point.tangent
            x0 = np.append(xs_time[-1][:, 0], branch_point.F)
            try:
                # HBM case
                x0_tng = np.append(fourier.inv_DFT(x_tng[:-1])[:, 0], x_tng[-1])
            except:
                # Shooting case
                x0_tng = x_tng
            ax.plot(*zip(x0, x0_tng), "gray")

            if branch_point.F < F_range[0] or branch_point.F > F_range[1]:
                break

        x0s = np.array([x_time[:, 0] for x_time in xs_time])
        amplitudes = np.array(amplitudes)
        stable = np.array(stable)
        unstable = np.invert(stable)
        Fs = np.array(Fs)
        F_stable = Fs.copy()
        F_stable[unstable] = np.nan
        F_unstable = Fs.copy()
        F_unstable[stable] = np.nan

        ax.plot(x0s[:, 0], x0s[:, 1], np.squeeze(Fs), linewidth=1)

        plt.figure(fig_stability)

        plt.plot(F_stable, amplitudes, label="stable")
        plt.plot(F_unstable, amplitudes, "--", label="unstable")
        # plt.title(f"Amplitude of first DOF -- {solver}")
    plt.legend()


def test_bifurcation_curve_comparison(hbm_system, shooting_system):

    if hbm_system.equations[0].ode.autonomous:
        parameter = "nu"
        param_range = (0, 3.5)
    else:
        parameter = "F"
        param_range = (0, 20)

    solver = NewtonSolver()

    branches = [
        run_continuation(system, solver, parameter, param_range, max_steps=200)
        for system in [hbm_system, shooting_system]
    ]

    # Compare the curves using interpolation
    idx_prev = 0
    for k, bp_shoot in enumerate(branches[1][:-1]):
        if not param_range[0] <= getattr(bp_shoot, parameter) <= param_range[1]:
            print(f"{k}-th shooting point outside {parameter} range")
            continue
        print(f"k = {k}: {parameter} = {getattr(bp_shoot, parameter)}, d = ", end="")
        d, idx_prev = interpolate_bp(idx_prev, branches[0], bp_shoot)
        print(f"{d}, idx={idx_prev}")

        assert d < 1e-2
    pass


def interpolate_bp(idx_prev, branch_hbm, bp_shoot):

    if idx_prev >= 10:
        pass

    if not 0 <= idx_prev < len(branch_hbm) - 1:
        raise ValueError(f"Index {idx_prev} is outside branch")

    x_shoot = bp_shoot.vector_of_unknowns
    param = bp_shoot.unknowns[-1]

    for k in (0, 1):
        if branch_hbm[idx_prev + k].equations[0].ode.autonomous:
            # account for freedom of phase: interpolate over x(t) to find value closest to x
            x_t = branch_hbm[idx_prev + k].equations[0].x_time()
            x = interpolate_time(int(np.ceil(x_t.shape[1] / 2)), x_t, x_shoot[:2])
            x = np.append(x, 2 * np.pi / branch_hbm[idx_prev + k].omega)
        else:
            x = branch_hbm[idx_prev + k].equations[0].x_time()[:, 0]

        x = np.append(x, getattr(branch_hbm[idx_prev + k], param))
        x = np.real(x)

        if k == 0:
            x_prev = x
        else:
            x_next = x

    d_shoot = x_shoot - x_prev
    d_hbm = x_next - x_prev

    alpha = np.inner(d_shoot, d_hbm) / np.linalg.norm(d_hbm) ** 2
    if alpha < -1e-3:
        return interpolate_bp(int(idx_prev - np.ceil(-alpha)), branch_hbm, bp_shoot)
    elif alpha > 1 + 1e-3:
        return interpolate_bp(int(idx_prev + np.floor(alpha)), branch_hbm, bp_shoot)
    else:
        x_interp = x_prev + alpha * d_hbm
        return np.linalg.norm(x_shoot - x_interp), idx_prev


def interpolate_time(idx_prev, x_time, x_comp):
    d_comp = x_comp - x_time[:, idx_prev]
    d_time = x_time[:, np.mod(idx_prev + 1, x_time.shape[1])] - x_time[:, idx_prev]

    alpha = np.inner(d_comp, d_time) / np.linalg.norm(d_time) ** 2
    if alpha < -1e-3:
        idx_next = int(np.mod(idx_prev - np.ceil(-alpha), x_time.shape[1]))
        return interpolate_time(idx_next, x_time, x_comp)
    elif alpha > 1 + 1e-3:
        idx_next = int(np.mod(idx_prev + np.floor(alpha), x_time.shape[1]))
        return interpolate_time(idx_next, x_time, x_comp)
    else:
        return x_time[:, idx_prev] + alpha * d_time


def bif_curve_aut(sparse=False, real_formulation=False):
    nu_0 = 0.1
    omega_0 = 1
    N_HBM = 45
    L_DFT = 1000

    print("Van der Pol oscillator")

    fourier = Fourier(
        N_HBM=N_HBM,
        L_DFT=L_DFT,
        n_dof=2,
        sparse_formulation=sparse,
        real_formulation=real_formulation,
    )

    ts = fourier.time_samples(omega_0)
    x0_samples = np.vstack(
        (2 * np.cos(omega_0 * ts), -2 * omega_0 * np.sin(omega_0 * ts))
    )
    X0 = fourier.DFT(x0_samples)

    stability_method = KoopmanHillSubharmonic(
        fourier=fourier, autonomous=True, tol=1e-4
    )

    initial_problem = HBMProblem_autonomous(
        f=vanderpol,
        initial_guess=X0,
        omega=omega_0,
        fourier=fourier,
        variable="x",
        stability_method=stability_method,
        verbose=False,
        parameters_f={"nu": nu_0},
    )

    initial_problem.solve()
    print(f"Initial problem convergence: {initial_problem.converged}")

    nu_range = (nu_0, 10)

    xs_time = []
    amplitudes = []
    nus = []
    stable = []
    omegas = []
    floquet_multipliers = []
    x0s = []

    for branch_point in pseudo_arclength_continuator(
        initial_problem=initial_problem,
        stepsize=0.1,
        stepsize_range=(0.001, 0.15),
        initial_direction=1,
        num_steps=1000,
        key_param="nu",
        value_param=nu_0,
        verbose=True,
    ):
        xs_time.append(branch_point.x_time())
        amplitudes.append(np.max(xs_time[-1][0, :]))
        nus.append(branch_point.nu)
        stable.append(branch_point.stable)
        omegas.append(branch_point.omega)
        floquet_multipliers.append(branch_point.eigenvalues)

        x0 = xs_time[-1][:, 0]

        if branch_point.nu < nu_range[0] or branch_point.nu > nu_range[1]:
            break

    amplitudes = np.array(amplitudes)
    stable = np.array(stable)
    unstable = np.invert(stable)
    nus = np.array(nus)
    nu_stable = nus.copy()
    nu_stable[unstable] = np.nan
    nu_unstable = nus.copy()
    nu_unstable[stable] = np.nan

    plt.plot(nu_stable, amplitudes, "r+-", label="stable")
    plt.plot(nu_unstable, amplitudes, "bx-", label="unstable")
    # plt.title(f"Amplitude of first DOF -- {solver}")
    plt.legend()
    plt.xlabel("\nu")
    plt.ylabel("|x_1|")

    plt.figure()
    plt.plot(nu_stable, omegas, "r", label="stable")
    plt.plot(nu_unstable, omegas, "b", label="unstable")
    plt.xlabel("\nu")
    plt.ylabel("\omega")

    fig_anim, axs = plt.subplots(nrows=1, ncols=2)
    # Create an animation
    (line2,) = axs[0].plot(xs_time[0][0, :], xs_time[0][1, :])
    axs[0].axis("equal")
    (plot_floquet_multiplier,) = axs[1].plot(
        np.real(floquet_multipliers[0]),
        np.imag(floquet_multipliers[0]),
        "x",
    )
    phis = np.linspace(0, 2 * np.pi)
    axs[1].plot(np.cos(phis), np.sin(phis), "k-")
    axs[1].axis("equal")

    def update(frame):
        line2.set_xdata(xs_time[frame][0, :])
        line2.set_ydata(xs_time[frame][1, :])
        plot_floquet_multiplier.set_xdata(np.real(floquet_multipliers[frame]))
        plot_floquet_multiplier.set_ydata(np.imag(floquet_multipliers[frame]))
        return (line2, plot_floquet_multiplier)

    animation = FuncAnimation(
        fig=fig_anim, func=update, frames=len(xs_time), interval=20
    )
    return animation


if __name__ == "__main__":
    bif_curve_nonaut(real_formulation=True)
    # plt.figure()
    # animation = bif_curve_aut(real_formulation=True, sparse=False)
    plt.show()
    print("All tests successful.")
