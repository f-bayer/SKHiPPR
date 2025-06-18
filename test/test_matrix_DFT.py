import numpy as np
from skhippr.Fourier import Fourier
import pytest


def test_matrix_DFT_and_iDFT(N_HBM=6, L_DFT=16, real_formulation=False):
    fourier = Fourier(
        N_HBM=N_HBM,
        L_DFT=L_DFT,
        n_dof=2,
        real_formulation=real_formulation,
    )
    omega = np.random.rand()
    phase_3 = np.random.rand()

    # Check minimum required frequency
    assert N_HBM >= 5
    ts = fourier.time_samples(omega)[np.newaxis, np.newaxis, :]

    # Construct test case
    A_samples = np.vstack(
        (
            np.hstack(
                (
                    3 * np.cos(omega * ts + 3),
                    -np.sin(omega * ts) + 0.5 * np.cos(5 * omega * ts + phase_3),
                )
            ),
            np.hstack((-4 * np.sin(2 * omega * ts) + 3, np.ones((1, 1, fourier.L_DFT)))),
        )
    )

    # Compare transformations
    A = fourier.matrix_DFT(A_samples)
    A_old = fourier._matrix_DFT(A_samples)

    assert np.allclose(A, A_old, atol=1e-14, rtol=1e-14)

    # Compare inverse transformations
    A_samples_fft = fourier.matrix_inv_DFT(A)
    assert np.allclose(A_samples_fft, A_samples, atol=1e-14, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__])
