import pytest
import numpy as np
from copy import copy, replace

from skhippr.Fourier import Fourier
from skhippr.odes.nonautonomous import Duffing
from skhippr.problems.HBM import HBMEquation
from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillSubharmonic,
)


@pytest.fixture
def ode():
    return Duffing(t=0, x=np.array([0, 1]), alpha=1, beta=3, F=1, delta=0.1, omega=1.3)


def test_C_W_construction(fourier_small, verbose=False):
    if verbose:
        print(fourier_small)

    KHP_subh = KoopmanHillSubharmonic(fourier=fourier_small)
    KHP_tilde = KHP_double_size(fourier_small)

    # Check W
    W_ref, W_subh_ref = extract_blocks_CW(KHP_tilde.W, KHP_tilde.fourier.n_dof)
    assert (W_ref == KHP_subh.W).all(), f" W ref: {W_ref} \n W: {KHP_subh.W}"
    assert (
        W_subh_ref == KHP_subh.W_subh
    ).all(), f" W_subh ref: {W_subh_ref} \n W_subh: {KHP_subh.W_subh}"

    # Check C
    C_ref, C_subh_ref = extract_blocks_CW(KHP_tilde.C, KHP_tilde.fourier.n_dof)
    assert (C_ref == KHP_subh.C).all()
    assert (C_subh_ref == KHP_subh.C_subh).all()


@pytest.mark.parametrize("t_over_period", [1, 2, 5, 0.2, 0.5, np.sqrt(2) / 2])
def test_C_subh_time(fourier_small, t_over_period):
    KHP_subh = KoopmanHillSubharmonic(fourier=fourier_small)
    KHP_tilde = KHP_double_size(fourier_small)

    C_time_tilde = KHP_tilde.C_time(0.5 * t_over_period)  # T_tilde = 2*T
    [C_ref, C_subh] = extract_blocks_CW(C_time_tilde, KHP_tilde.fourier.n_dof)

    assert np.allclose(KHP_subh.C_time(t_over_period), C_ref)
    assert np.allclose(KHP_subh.C_subh_time(t_over_period), C_subh)


def test_subharmonic_hill_matrix(fourier, solver, ode):
    """Compute periodic solution using HBM"""

    ts = fourier.time_samples(ode.omega)
    x0 = np.vstack([np.cos(ode.omega * ts), -ode.omega * np.sin(ode.omega * ts)])
    hbm = HBMEquation(ode, ode.omega, fourier, fourier.DFT(x0))

    solver.solve_equation(hbm, "X")

    H = hbm.hill_matrix()
    KHP_subh = KoopmanHillSubharmonic(fourier)
    H_subh = KHP_subh.hill_subh(hbm)

    # Solve period-doubled problem
    fourier_subh = Fourier(
        2 * fourier.N_HBM, 2 * fourier.L_DFT, fourier.n_dof, fourier.real_formulation
    )
    x_time = hbm.x_time()
    X0_subh = fourier_subh.DFT(np.hstack((x_time, x_time)))

    hbm_subh = HBMEquation(ode, ode.omega, fourier_subh, X0_subh, period_k=2)

    solver.solve_equation(hbm_subh, "X")
    assert np.allclose(X0_subh, hbm_subh.X, atol=1e-5)

    H_tilde = hbm_subh.hill_matrix()

    H_ref, H_sub_ref = extract_blocks_H(H_tilde, fourier_subh.n_dof)

    assert np.allclose(H_ref, b=H, atol=2e-5, rtol=2e-5)
    assert np.allclose(H_sub_ref, H_subh, atol=2e-5, rtol=2e-5)


def KHP_double_size(fourier, real_formulation=None):
    """Create the subharmonic Koopman-Hill projection "manually"
    by doubling the size of the DFT and changing C"""

    if real_formulation is None:
        real_formulation = fourier.real_formulation

    # Fourier formulation of twice the length with real_formulation set explicitly
    fourier_tilde = replace(
        fourier, N_HBM=2 * fourier.N_HBM, real_formulation=real_formulation
    )
    KHP_tilde = KoopmanHillProjection(fourier=fourier_tilde)

    # Change the C matrix of KHP_tilde to (I, -I, I, ..., -I, I)
    C0 = np.ones((1, 2 * fourier_tilde.N_HBM + 1))
    C0[:, 1::2] = -1
    C0 = np.kron(C0, np.eye(fourier_tilde.n_dof))

    # Adjust for real formulation
    if real_formulation:
        C0 = C0 @ fourier_tilde.T_to_cplx_from_real

    KHP_tilde.C = C0
    return KHP_tilde


def extract_blocks_CW(matrix_tilde, n_dof):
    """Extract matrix_dir (even blocks; C or W)
    and matrix_subh (uneven blocks; C_subh or W_subh)
    from matrix_tilde (C_tilde or W_tilde)"""
    if matrix_tilde.shape[1] > n_dof:
        transpose = True
        matrix_tilde = matrix_tilde.T
    else:
        transpose = False

    blocks = matrix_tilde.reshape(-1, n_dof, n_dof, order="C")
    # k-th block is at blocks[k, :, :]

    matrix_dir = blocks[0::2, :, :].reshape(-1, n_dof, order="C")
    matrix_subh = blocks[1::2, :, :].reshape(-1, n_dof, order="C")

    if transpose:
        matrix_dir = matrix_dir.T
        matrix_subh = matrix_subh.T

    return matrix_dir, matrix_subh


def extract_blocks_H(matrix, n_dof):
    """Extract the uneven and even n_dof-blocks of matrix by similarity transform."""
    size_smaller_block = int(0.5 * (matrix.shape[0] / n_dof - 1))
    E0 = np.eye(2 * size_smaller_block + 1)
    M0 = np.vstack([E0[::2, :], E0[1::2, :]])
    M = np.kron(M0, np.eye(n_dof))

    matrix_blocks = M @ matrix @ M.T
    idx_split = [
        0,
        n_dof * (size_smaller_block + 1),
        n_dof * (2 * size_smaller_block + 1),
    ]
    blocks = []
    for k in range(len(idx_split) - 1):
        blocks.append(
            [
                matrix_blocks[
                    idx_split[k] : idx_split[k + 1], idx_split[l] : idx_split[l + 1]
                ]
                for l in range(len(idx_split) - 1)
            ]
        )

    if np.max(np.abs(blocks[0][1])) > 1e-10 or np.max(np.abs(blocks[1][0])) > 1e-10:
        raise ValueError(
            "Matrix can not be block diagonalized by even/odd column resorting!"
        )

    return blocks[0][0], blocks[1][1]


if __name__ == "__main__":
    pytest.main([__file__])
