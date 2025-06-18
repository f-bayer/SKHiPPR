import pytest
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from skhippr.systems.autonomous import vanderpol
from skhippr.systems.nonautonomous import duffing

from skhippr.problems.shooting import ShootingProblem


@pytest.mark.parametrize(
    "period_k,unpack_parameters",
    [
        (1, True),
        (1, False),
        (5, True),
        (5, False),
    ],
)
def test_shooting_duffing(params_duffing, period_k, unpack_parameters, visualize=False):

    params = params_duffing[period_k]
    omega = params["omega"]
    if unpack_parameters:
        alpha = params["alpha"]
        beta = params["beta"]
        F = params["F"]
        delta = params["delta"]

        f = lambda t, x: duffing(t, x, omega, alpha, beta, F, delta)
        params = dict()
    else:
        f = duffing

    T = 2 * np.pi * period_k / omega
    ode_kwargs = {"rtol": 1e-7, "atol": 1e-7}

    f_ivp = lambda t, x: f(t, x, **params)[0]

    x0 = np.array([1.0, 0.0])

    # Converge to periodic solution by brute-force integration
    num_periods = 100
    sol_ivp_converge = solve_ivp(f_ivp, (0, num_periods * T), x0, **ode_kwargs)
    x0_p = sol_ivp_converge.y[:, -1]
    # Ensure that solution didn't diverge
    assert all(
        np.abs(x0_p) < 100
    ), f"Trajectory from initial condition diverged: {x0_p}"

    # Find and plot converged solution by integrating over period_k periods
    points_per_period = 300
    t_eval = np.linspace(0, T, period_k * points_per_period + 1, endpoint=True)
    sol_ivp = solve_ivp(f_ivp, (0, T), x0_p, t_eval=t_eval, **ode_kwargs)
    assert np.allclose(
        x0_p, sol_ivp.y[:, -1], atol=1e-2, rtol=1e-2
    ), "Brute-force integration did not converge to periodic solution"
    if visualize:
        plt.figure()
        plt.plot(
            sol_ivp.y[0, :],
            sol_ivp.y[1, :],
            label=f"sol after {num_periods} periods",
        )
        plt.plot(x0_p[0], x0_p[1], "+", label="Starting point")
        plt.title(f"Duffing period-{period_k} solution")

    # Find periodic solution using shooting
    prb = ShootingProblem(
        f=f,
        x0=x0,
        T=T,
        autonomous=False,
        variable="x",
        verbose=visualize,
        kwargs_odesolver=ode_kwargs,
        parameters=params,
        period_k=period_k,
    )
    if visualize:
        print(prb)

    prb.solve()

    if visualize:
        print(prb)

    assert prb.converged, f"Shooting method did not converge"
    assert np.allclose(
        prb.x, x0_p, atol=1e-2, rtol=1e-2
    ), f"Shooting method did not converge to attractive periodic solution"
    assert prb.stable, "Shooting: Wrong stability assertion"

    x_time = prb.x_time(t_eval)
    assert np.allclose(
        prb.x, x_time[:, -1], atol=1e-5, rtol=1e-5
    ), f"Shooting method did not converge to periodic solution"
    assert np.allclose(
        x_time, sol_ivp.y, atol=1e-5, rtol=1e-5
    ), f"Shooting method time series does not match converged solution"

    # Verify that solution is indeed a period-k solution
    for i in range(1, period_k):
        assert not np.allclose(
            x_time[:, i * points_per_period],
            x_time[:, 0],
            atol=1e-5,
            rtol=1e-5,
        ), f"Shooting solution is not period-{period_k} but period-{i}"

    if visualize:
        plt.plot(x_time[0, :], x_time[1, :], "r--", label="Shooting solution")
        plt.legend()


def test_shooting_vanderpol():

    nu = 0.5
    omega0 = 1

    f_ivp = lambda t, x: vanderpol(t, x, nu)[0]
    x0 = np.array([2, 0])

    ode_kwargs = {"rtol": 1e-7, "atol": 1e-7}
    parameters = {"nu": nu}

    # Find periodic solution
    prb = ShootingProblem(
        f=vanderpol,
        x0=x0,
        T=2 * np.pi / omega0,
        autonomous=True,
        variable="x",
        verbose=True,
        kwargs_odesolver=ode_kwargs,
        parameters=parameters,
    )
    print(prb)
    prb.solve()
    print(prb)
    assert prb.converged

    # Verify that it is indeed a periodic solution
    x_time = prb.x_time()
    assert np.allclose(prb.x[:2], x_time[:, -1], rtol=1e-3, atol=1e-3)

    # Verify stability:
    sol_ivp_converge = solve_ivp(f_ivp, (0, 300 * prb.T), x0, **ode_kwargs)
    x_end = sol_ivp_converge.y[:, -1]
    assert prb.stable == np.allclose(x_end, prb.x[:2], atol=1e-1, rtol=1e-1)

    # plot problem
    # plt.figure()
    # plt.plot(x_time[0, :], x_time[1, :])
    # plt.title("Van der Pol solution")


if __name__ == "__main__":
    pytest.main([__file__])
