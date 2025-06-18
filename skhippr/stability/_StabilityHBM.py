from abc import ABC, abstractmethod
import numpy as np
import warnings

from skhippr.Fourier import Fourier
from skhippr.stability._StabilityMethod import _StabilityMethod
from skhippr.problems.HBM import HBMProblem


class _StabilityHBM(_StabilityMethod):
    """
    Abstract base class for stability methods for periodic solutions, applicable to HBM problems.

    In addition to the methods inherited from :py:class:`~skhippr.stability._StabilityMethod._StabilityMethod`, it requires a method to compute the fundamental solution matrix of the periodic solution.
    With this, the Floquet multipliers (eigenvalues), which govern stability based on their magnitude, can be determined.


    Attributes:
    -----------

    fourier : :py:class:`~skhippr.Fourier.Fourier`
        :py:class:`~skhippr.Fourier.Fourier` object representing the FFT configuration (in particular: ``N_HBM`` and real/complex formulation) of the considered problem class.
    tol : float
        Tolerance for stability checks.
    autonomous : bool, optional
        Indicates if the system is autonomous and has one Floquet multiplier at ``1`` due to the freedom of phase. Default: ``False``.

    """

    def __init__(
        self, label: str, fourier: Fourier, tol: float, autonomous: bool = False
    ):
        super().__init__(label=label, tol=tol)
        self.fourier = fourier
        self.autonomous = autonomous

    @abstractmethod
    def fundamental_matrix(
        self, t_over_period: float, problem: HBMProblem
    ) -> np.ndarray:
        """
        Compute the fundamental matrix at a given normalized time for a specified periodic solution.

        Parameters
        ----------

        t_over_period : float
            Normalized time over the period (typically between 0 and 1).
        problem : :py:class:`~skhippr.problems.HBM.HBMProblem`
            The (solved) :py:class:`~skhippr.problems.HBM.HBMProblem` of whose solution the fundamental matrix is sought.

        Returns
        -------
        np.ndarray
            The computed fundamental matrix as a NumPy array.
        """
        ...

    def determine_eigenvalues(self, problem: HBMProblem) -> np.ndarray:  # type:ignore
        """
        Determine the Floquet multipliers of the periodic solution (eigenvalues of the monodromy matrix) for the periodic solution encoded in the given :py:class:`~skhippr.problems.HBM.HBMProblem`.

        Parameters
        ----------

        problem : :py:class:`~skhippr.problems.HBM.HBMProblem`
            The (solved) harmonic balance method (HBM) problem of whose solution the Floquet multipliers are sought.

        Returns
        -------

        np.ndarray
            1-D Array of Floquet multipliers (eigenvalues of the monodromy matrix).

        """

        monodromy = self.fundamental_matrix(t_over_period=1, problem=problem)
        floquet_multipliers = np.linalg.eigvals(monodromy)
        return floquet_multipliers

    def determine_stability(self, eigenvalues) -> bool:
        """
        Determines the stability of a periodic solution based on the Floquet multipliers (``eigenvalues``).

        The periodic solution is asserted stable if all Floquet multipliers lie within the unit circle.

        In the autonomous case, one Floquet multiplier at (numerically) ``1`` is expected, irrespective of the stability properties (due to the freedom of phase).
        Therefore, if ``self.autonomous`` is ``True``, the method identifies and excludes the Floquet multiplier closest to ``1``.
        If this multiplier deviates from ``1`` by more than ``self.tol``, a warning is issued.

        Parameters
        ----------

        eigenvalues : array_like
            1-D Array of Floquet multipliers (eigenvalues) of the periodic solution.

        Returns
        -------

        bool
            ``True`` if the periodic solution is stable, ``False`` otherwise.

        Warns
        -----

        UserWarning
            In the autonomous case, if the Floquet multiplier associated with the freedom of phase is not sufficiently close to ``1``.
            This indicates that not enough harmonics were used to accurately cover the nonlinearities of the system.
        """

        if self.autonomous:
            idx_freedom_of_phase = np.argmin(np.abs(eigenvalues - 1))
            freedom_of_phase = eigenvalues[idx_freedom_of_phase]
            eigenvalues = np.delete(eigenvalues, idx_freedom_of_phase)

            if abs(freedom_of_phase - 1) > self.tol:
                warnings.warn(
                    f"Floquet multiplier {freedom_of_phase} does not satisfy freedom of phase! "
                )

        return np.all(np.abs(eigenvalues) < 1 + self.tol)  # type:ignore

    def error_bound(self, t, a, b):
        """
        Compute an error bound for the fundamental solution matrix based on the exponential decay of Fourier coefficient matrices as returned by :py:class:`~skhippr.problems.HBM.HBMProblem.exponential_decay_parameters`.

        ``a`` and ``b`` bound the norm of the Fourier coefficients by::

            || J_k || <= a*np.exp(-b*np.abs(k)).

        Parameters
        ----------

        t : float
            The time at which the error bound is evaluated.
        a : float
            Factor in front of the exponential decay
        b : float
            Exponential decay

        Returns
        -------

        float
            The computed error bound for the fundamental solution matrix.

        Raises
        ------

        AttributeError
            If the stability method does not provide an error bound.

        """
        raise AttributeError(
            f"Stability method {self.__class__} provides no error bound."
        )

    def __str__(self):
        return f"{self.label} for {self.fourier}"
