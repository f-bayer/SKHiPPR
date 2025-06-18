import pytest
import numpy as np

from skhippr.Fourier import Fourier
from skhippr.stability.KoopmanHillProjection import KoopmanHillProjection


def test_T_cr_rc(fourier_small):
    """Test that the transformation matrices T_cr and T_rc are indeed inverses of each other"""

    T_rc_cr = fourier_small.T_to_real_from_cplx @ fourier_small.T_to_cplx_from_real
    T_cr_rc = fourier_small.T_to_cplx_from_real @ fourier_small.T_to_real_from_cplx

    for mat in (T_rc_cr, T_cr_rc):
        assert mat.shape == ((2 * fourier_small.N_HBM + 1) * fourier_small.n_dof,) * 2

        eye = np.eye(fourier_small.T_to_cplx_from_real.shape[0])
        assert np.allclose(mat, eye, atol=1e-10)


@pytest.mark.parametrize("t_over_period", [1, 2, 0.5, 0.2, np.sqrt(2) / 2])
def test_D_matrix(t_over_period):
    N_HBM = 5
    L_DFT = 20
    fourier_real = Fourier(N_HBM=N_HBM, L_DFT=L_DFT, n_dof=2, real_formulation=True)
    fourier_cplx = fourier_real.__replace__(real_formulation=False)

    KHP_real = KoopmanHillProjection(fourier=fourier_real)
    KHP_cplx = KoopmanHillProjection(fourier=fourier_cplx)

    assert np.allclose(
        fourier_real.T_to_cplx_from_real, fourier_cplx.T_to_cplx_from_real
    )
    assert np.allclose(
        fourier_real.T_to_real_from_cplx, fourier_cplx.T_to_real_from_cplx
    )

    D_real = KHP_real.D_time(t_over_period)
    D_cplx = KHP_cplx.D_time(t_over_period)

    D_real_trafo = (
        fourier_real.T_to_real_from_cplx @ D_cplx @ fourier_real.T_to_cplx_from_real
    )
    D_cplx_trafo = (
        fourier_cplx.T_to_cplx_from_real @ D_real @ fourier_cplx.T_to_real_from_cplx
    )
    assert np.allclose(
        D_real, D_real_trafo, atol=1e-10
    ), f"Max D error: {np.max(np.abs(D_real - D_real_trafo))}"
    assert np.allclose(
        D_cplx, D_cplx_trafo, atol=1e-10
    ), f"Max D error: {np.max(np.abs(D_cplx - D_cplx_trafo))}"


@pytest.mark.parametrize("t_over_period", [1, 2, 5, 0.5, 0.2, np.sqrt(2)])
def test_C_time(fourier_small, t_over_period):
    """Test that the C_time function is correct for real and complex formulation."""

    omega = 0.35
    KHP = KoopmanHillProjection(fourier=fourier_small)

    # Override C with nontrivial values
    C_cplx = np.random.rand(*KHP.C.shape)
    if fourier_small.real_formulation:
        C_0 = C_cplx @ KHP.fourier.T_to_cplx_from_real
    else:
        C_0 = C_cplx
    KHP.C = C_0

    C_time = KHP.C_time(t_over_period)

    # Determine C_cplx(t) as reference
    t = t_over_period * 2 * np.pi / omega
    exp_vals = 1j * omega * t * np.arange(-fourier_small.N_HBM, fourier_small.N_HBM + 1)
    D_vals = np.exp(np.kron(exp_vals, np.ones(fourier_small.n_dof)))
    C_ref = C_cplx * D_vals[np.newaxis, :]

    # Adjust for real formulation
    if fourier_small.real_formulation:
        C_ref = C_ref @ KHP.fourier.T_to_cplx_from_real

    assert (
        C_time.shape == C_ref.shape
    ), f"C_time shape is {C_time.shape} while C_ref is shape {C_ref.shape}"
    assert np.allclose(
        C_time, C_ref, atol=1e-10
    ), f"Max C error: {np.max(np.abs(C_time - C_ref))}"


if __name__ == "__main__":
    pytest.main([__file__])
