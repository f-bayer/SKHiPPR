import pytest
import numpy as np

from skhippr.Fourier import Fourier


@pytest.fixture
def params_duffing():
    return {
        1: {
            "alpha": 1,
            "beta": 3,
            "F": 1,
            "delta": 0.1,
            "omega": 1.3,
        },
        5: {
            "alpha": -1,
            "beta": 1,
            "F": 0.37,
            "delta": 0.3,
            "omega": 1.2,
        },
    }


@pytest.fixture(
    scope="module",
    params=[
        {"real_formulation": True},
        {"real_formulation": False},
    ],
)
def fourier_small(request):
    # warnings.filterwarnings(action="error", category=np.exceptions.ComplexWarning)
    n_dof = 2
    N_HBM = 3
    L_DFT = 16

    return Fourier(
        N_HBM=N_HBM,
        L_DFT=L_DFT,
        n_dof=n_dof,
        real_formulation=request.param["real_formulation"],
    )


@pytest.fixture(
    scope="module",
    params=[
        {"real_formulation": True},
        {"real_formulation": False},
    ],
)
def fourier(request):
    # warnings.filterwarnings(action="error", category=np.exceptions.ComplexWarning)
    n_dof = 2
    N_HBM = 50
    L_DFT = 400

    return Fourier(
        N_HBM=N_HBM,
        L_DFT=L_DFT,
        n_dof=n_dof,
        real_formulation=request.param["real_formulation"],
    )
