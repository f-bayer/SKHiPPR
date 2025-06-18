import pytest
import numpy as np
from skhippr.systems.autonomous import blockonbelt


@pytest.fixture
def blockonbelt_params():
    epsilon = 0.1
    k = 0.5
    m = 1.0
    Fs = 0.5
    vdr = 0.1
    delta = 0.1

    return epsilon, k, m, Fs, vdr, delta


def test_blockonbelt_array_inputs(blockonbelt_params):
    """
    Test the blockonbelt function when t is an array and x is a 2xL array.
    """
    t = np.linspace(0, 10, 100)
    x = np.random.rand(2, len(t))

    f, derivatives = blockonbelt(t, x, *blockonbelt_params)

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


def test_blockonbelt_scalar_inputs(blockonbelt_params):
    """
    Test the blockonbelt function when t is a float and x is a 1-dimensional array of length 2.
    """
    t = 1.0
    x = np.random.rand(2)

    f, derivatives = blockonbelt(t, x, *blockonbelt_params)

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
