"""
This script demonstrates the effect of aliasing error on the accuracy of the Koopman-Hill projection (KHP) for the Meissner equation by varying the Discrete Fourier Transform (DFT) length. It compares the accuracy of computed Floquet multipliers for different DFT lengths and visualizes the results.
Functions:
    analyze_N_meissner_aliasing(N_max=60, Ls_DFT=(1024,), csv_path=None):
        Analyzes and visualizes the impact of DFT length (L_DFT) on the accuracy of the Koopman-Hill projection for the Meissner equation. Optionally saves results to a CSV file.
Usage:
    Run the script directly to generate plots and optionally save results to a CSV file. The main function `analyze_N_meissner_aliasing` is called with example parameters to showcase the influence of aliasing on prediction accuracy.
    - The Meissner equation's Fourier coefficients decay as O(1/|k|), with every even coefficient being zero and odd ones alternating in sign.
    - Aliasing errors affect the accuracy of the computed Floquet multipliers.
    - Knowledge about the sign of the (true) Fourier coefficients can significally improve the results -in the case of Meissner choosing a DFT length that is divisible by 2 but not by 4
"""

import numpy as np
import matplotlib.pyplot as plt
from skhippr.odes.ltp import (
    meissner,
    meissner_fourier,
    meissner_fundamental_matrix,
    meissner_g,
)

from demo_mathieu_N_convergence import (
    analyze_N_convergence,
    initialize_csv,
    setup_plot,
)


def analyze_N_meissner_aliasing(N_max=60, Ls_DFT=(1024,), csv_path=None):
    """
    Showcase the effect of aliasing error on KHP accuracy for the Meissner equation by varying the DFT length.
    See notes for an explanation of the observations.

    This function initializes a CSV file for results, sets up system parameters, computes the reference monodromy matrix,
    and iterates over different DFT lengths to analyze the impact on the accuracy of the computed Floquet multipliers.
    Results are plotted and optionally saved to a CSV file.
    Args:
        N_max (int, optional): Maximum order of the Hill determinant to use. Default is 60.
        Ls_DFT (tuple of int, optional): Tuple of DFT lengths to analyze. Default is (1024,).
        csv_path (str or None, optional): Path to the CSV file for saving results. If None, results are not saved.
    Returns:
        None
    Notes:
        The Fourier coefficients of the Meissner equation decay with O(1/|k|).
        Every even Fourier coefficient is zero, and the odd ones have alternating sign.

        When  L_DFT is approximately 1e3, the aliasing errors in the Fourier coefficients are approx. 1e-3.
        Hence, the error incurred by aliasing can be at most approx. 1e-3 (case L_DFT=125).
        If L_DFT is uneven, nonzero Fourier coefficients with k > L_DFT are added as aliasing error
        to the even Fourier coefficients which are supposed to be zero, leading to the largest possible error.
        If L_DFT is divisible by 4, positive Fourier coefficients with k > L_DFT are added as aliasing error to positive Fourier coefficients,
        and vice versa. Hence, the total aliasing error is given by a series whose summands all have the same sign (case L_DFT = 1024).
        In contrast, if L_DFT is divisible by 2 but not by 4, the series for the total aliasing error is alternating in sign, making the resulting aliasing error negligible (case L_DFT=1026).
    """
    initialize_csv(csv_path, N_max, key_param="L_DFT")
    params = {"a": 4, "b": 0.2, "omega": 1, "d": 0.0}
    params_plot = {}
    T = 2 * np.pi / params["omega"]
    Phi_T_ref = meissner_fundamental_matrix(T, 0, **params)
    lambdas_ref = np.linalg.eig(Phi_T_ref).eigenvalues
    ax_conv, axs, _ = setup_plot(None, lambdas_ref)
    ax_time = axs[1]
    ax_time.clear()

    # Plot true square function into right figure
    ts = np.linspace(0, T, max(Ls_DFT) + 1, endpoint=True)
    g, _ = meissner_g(ts, params["a"], params["b"], params["omega"])
    ax_time.plot(ts, g, "k", label="True square function")
    ax_time.set_xlim((0.23 * T, 0.27 * T))
    ax_time.set_ylim((params["a"] - 1.2 * params["b"], params["a"] + 1.2 * params["b"]))
    # Perform Koopman-Hill projection on Meissner equation for various L_DFT
    for k, L_DFT in enumerate(Ls_DFT):
        params_plot["color"] = f"C{k}"
        params_plot["label"] = f"L_DFT={L_DFT}"
        hbm = analyze_N_convergence(
            meissner,
            params,
            Phi_T_ref=Phi_T_ref,
            N_max=N_max,
            csv_path=csv_path,
            key_param="L_DFT",
            value_param=L_DFT,
            params_plot=params_plot,
            ax_conv=ax_conv,
        )

        # Plot the identified "square" function
        ts = np.linspace(0, T, L_DFT, endpoint=False)
        J_time = hbm.ode_samples()
        ax_time.plot(ts, -J_time[1, 0, :].squeeze(), "--", **params_plot)

    # Perform Koopman-Hill projection without any aliasing (i.e., on the function given by the truncated Fourier series of the square function)
    params_plot["color"] = f"C{k+1}"
    params_plot["marker"] = "o"
    params_plot["fillstyle"] = "none"
    params_plot["label"] = "No aliasing error"

    params["N_HBM"] = N_max
    analyze_N_convergence(
        meissner_fourier,
        params,
        Phi_T_ref=Phi_T_ref,
        N_max=N_max,
        csv_path=csv_path,
        key_param="L_DFT",
        value_param=L_DFT,
        params_plot=params_plot,
        ax_conv=ax_conv,
    )

    ax_conv.legend()
    ax_conv.set_title("Aliasing error in J_k affects KHP accuracy")
    ax_time.legend()


if __name__ == "__main__":
    # Showcase the influence of L_DFT (i.e., the influence of aliasing) on the prediction accuracy of the Meissner eq.

    analyze_N_meissner_aliasing(
        N_max=60,
        csv_path="data_meissner.csv",
        Ls_DFT=(1024, 1025, 1026, 1027, 1028, 1029, 1030),
    )
    plt.show()
