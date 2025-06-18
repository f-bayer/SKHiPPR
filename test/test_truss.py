import pytest
import numpy as np
from skhippr.systems.autonomous import truss


@pytest.fixture
def truss_params():
    k = 100.0
    c = 0.5
    F = 1.0
    a = 0.1
    l_0 = 1.0
    m = 1.0

    return k, c, F, a, l_0, m


def test_truss_array_inputs(truss_params):
    """
    Test the truss function when t is an array and x is a 2xL array.
    """
    t = np.linspace(0, 10, 100)
    x = np.random.rand(2, len(t))

    f, derivatives = truss(t, x, *truss_params)

    assert f.shape == x.shape, f"Expected f to have shape {x.shape}, but got {f.shape}"
    assert derivatives["x"].shape == (
        2,
        2,
        len(t),
    ), f"Expected df_dx to have shape {(2, 2, len(t))}, but got {derivatives['x'].shape}"
    for key in derivatives:
        if key == "x":
            continue
        assert (
            derivatives[key].shape == x.shape
        ), f"Expected df_d{key} to have shape {x.shape}, but got {derivatives[key].shape}"


def test_truss_scalar_inputs(truss_params):
    """
    Test the truss function when t is a float and x is a 1-dimensional array of length 2.
    """
    t = 1.0
    x = np.random.rand(2)

    f, derivatives = truss(t, x, *truss_params)

    assert f.shape == x.shape, f"Expected f to have shape {x.shape}, but got {f.shape}"
    assert derivatives["x"].shape == (
        2,
        2,
    ), f"Expected df_dx to have shape {(2, 2)}, but got {derivatives['x'].shape}"
    for key in derivatives:
        if key == "x":
            continue
        assert (
            derivatives[key].shape == x.shape
        ), f"Expected df_d{key} to have shape {x.shape}, but got {derivatives[key].shape}"


if __name__ == "__main__":
    pytest.main([__file__])
