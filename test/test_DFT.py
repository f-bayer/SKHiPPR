import pytest
import numpy as np
from skhippr.Fourier import Fourier


def test_dimensions(fourier_small: Fourier):
    """Test the dimensions of the DFT matrix."""

    assert fourier_small.DFT_matrix.shape == (
        fourier_small.n_dof * (2 * fourier_small.N_HBM + 1),
        fourier_small.n_dof * fourier_small.L_DFT,
    )

    assert fourier_small.iDFT_matrix.shape == (
        fourier_small.n_dof * fourier_small.L_DFT,
        fourier_small.n_dof * (2 * fourier_small.N_HBM + 1),
    )


def test_time_samples(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print("Test time samples:", end=" ")

    omega = np.random.rand()
    ts = fourier_small.time_samples(omega)
    assert len(ts) == fourier_small.L_DFT

    T = 2 * np.pi / omega
    dt = T / fourier_small.L_DFT
    assert np.allclose(ts, np.arange(0, T, dt), atol=2e-14)

    if verbose:
        print("success")


def test_time_samples_multiple_periods(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print("Test time samples (multiple periods):", end=" ")

    omega = np.random.rand()
    T = 2 * np.pi / omega
    dt = T / fourier_small.L_DFT

    for periods in (3, np.sqrt(2)):
        ts = fourier_small.time_samples(omega, periods=periods)
        assert ts[0] == 0.0
        assert np.allclose(ts[1:] - ts[:-1], dt, atol=1e-14)
        assert ts[-1] >= (T * periods - dt) - 1e-14
        assert ts[-1] < T * periods

    if verbose:
        print("success")


def test_reconstruct_X(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print("Generate and reconstruct a vector of Fourier coefficients:", end=" ")

    X = np.random.rand(fourier_small.n_dof * (2 * fourier_small.N_HBM + 1))
    x_samp = fourier_small.inv_DFT(X)
    x_samp_deprecated = fourier_small._inv_DFT(X)
    assert np.allclose(x_samp, x_samp_deprecated)

    X_reconstructed = fourier_small.DFT(x_samp)
    X_reconstructed_deprecated = fourier_small._DFT(x_samp)
    # fig, axs = plt.subplots(ncols=1, nrows=2)
    # axs[0].plot(np.real(X), "+")
    # axs[0].plot(np.real(X_reconstructed), "x")
    # plt.show()
    assert np.allclose(X, X_reconstructed)
    assert np.allclose(X, X_reconstructed_deprecated)
    if verbose:
        print("success")


def generate_x_samp(DFT, omega):
    ts = DFT.time_samples(omega)
    assert DFT.N_HBM >= 3  # x_samp(t) has 3 harmonics

    x_samp = np.vstack(
        (np.sin(omega * ts) + 4 * np.cos(3 * omega * ts), np.ones_like(ts))
    )

    x_derivative = np.vstack(
        (
            omega * np.cos(omega * ts) - 12 * omega * np.sin(3 * omega * ts),
            np.zeros_like(ts),
        )
    )

    return x_samp, x_derivative


def test_reconstruct_x_samp(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print(
            "Generate and reconstruct a vector of time samples within Nyquist freq:",
            end=" ",
        )
    omega = np.random.rand()
    ts = fourier_small.time_samples(omega)
    assert fourier_small.N_HBM >= 3  # Prevent aliasing

    omega = np.random.rand()
    x_samp, _ = generate_x_samp(fourier_small, omega)

    # Transform to frequency domain
    X = fourier_small.DFT(x_samp)
    X_deprecated = fourier_small._DFT(x_samp)
    assert np.allclose(X, X_deprecated)

    # back to freq domain
    x_samp_reconstructed = fourier_small.inv_DFT(X)
    x_samp_reconstructed_deprecated = fourier_small._inv_DFT(X)
    # fig, axs = plt.subplots(ncols=x_samp.shape[0], nrows=1)
    # for k in range(x_samp.shape[0]):
    #     axs[k].plot(x_samp[k, :])
    #     axs[k].plot(x_samp_reconstructed[k, :], "+")
    # plt.show()

    assert np.allclose(x_samp, x_samp_reconstructed)
    assert np.allclose(x_samp, x_samp_reconstructed_deprecated)
    if verbose:
        print("success")


def test_derivative(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print("Test derivative matrix:", end=" ")

    omega = np.random.rand()
    x_samp, x_deriv = generate_x_samp(fourier_small, omega)

    # Derivative function
    assert np.allclose(x_deriv, fourier_small.differentiate(x_samp, omega))

    # Derivative matrix
    X = fourier_small.DFT(x_samp)
    X_deriv = omega * fourier_small.derivative_matrix @ X
    x_deriv_reconstructed = fourier_small.inv_DFT(X_deriv)
    assert np.allclose(x_deriv, x_deriv_reconstructed)

    print("success")


def test_limit_frequency(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print("Add a frequency at N_HBM:", end=" ")

    omega = np.random.rand()
    ts = fourier_small.time_samples(omega)
    x_samp, _ = generate_x_samp(fourier_small, omega)
    x_samp[1, :] += 2 * np.sin(fourier_small.N_HBM * omega * ts + np.pi / 3)
    X = fourier_small.DFT(x_samp)
    x_samp_reconstructed = fourier_small.inv_DFT(X)
    assert np.allclose(x_samp, x_samp_reconstructed)
    if verbose:
        print("success")


def test_too_large_frequency(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print("Add a frequency larger than N_HBM:", end=" ")

    omega = np.random.rand()
    ts = fourier_small.time_samples(omega)
    x_samp, _ = generate_x_samp(fourier_small, omega)
    x_samp[1, :] += 2 * np.sin((fourier_small.N_HBM + 1) * omega * ts + np.pi / 3)
    X = fourier_small.DFT(x_samp)
    x_samp_reconstructed = fourier_small.inv_DFT(X)
    assert not np.allclose(x_samp, x_samp_reconstructed)
    if verbose:
        print("success (expected results to differ)")


if __name__ == "__main__":
    pytest.main([__file__])
