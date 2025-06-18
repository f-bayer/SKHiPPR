.. SKHiPPR documentation master file, created by
   sphinx-quickstart on Fri Jun 13 09:22:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SKHiPPR documentation
=====================

.. figure:: skhippr.png
   :width: 200
   :alt: SKHiPPR logo: A skipper on a boat riding the Duffing frequency response curve
   :align: center
      


SKHiPPR (/ˈski-pr/) is a Python toolbox with focus on **S**\ tability using the **K**\ oopman-\ **Hi**\ ll **P**\ rojection method for **P**\ eriodic solutions and **R**\ esonance curves.

   

.. 

   SKHiPPR is a continuation toolbox developed by Fabia Bayer as part of a research project in close cooperation with Remco Leine at the Institute for Nonlinear Mechanics, University of Stuttgart, Germany. The main focus of the project is the Koopman-based Hill stability method for periodic solutions.

   For more information about the Koopman-Hill projection method, please see the following references:

   * Bayer and Leine (2023): *Sorting-free Hill-based stability analysis of periodic solutions through Koopman analysis*. Nonlinear Dyn 111, 8439–8466, https://doi.org/10.1007/s11071-023-08247-7.
   * Bayer et al. (2024): *Koopman-Hill Stability Computation of Periodic Orbits in Polynomial Dynamical Systems Using a Real-Valued Quadratic Harmonic Balance Formulation*. International Journal of Non-Linear Mechanics, 167, 104894, https://doi.org/10.1016/j.ijnonlinmec.2024.104894.
   * Bayer and Leine (2025, preprint): *Explicit error bounds and guaranteed convergence of the Koopman-Hill projection stability method for linear time-periodic dynamics*, https://arxiv.org/abs/2503.21318 
   * Project website: https://www.inm.uni-stuttgart.de/research_nonlinear_mechanics/project_bayer/

This toolbox is object-oriented and modularized. It generates continuation curves with stability information by combining four modular aspects:

#. A user-defined system function returns the right-hand side of a system of (either algebraic or differential) equations as well as the derivatives.
#. A :py:class:`~skhippr.problems.newton.NewtonProblem` (or subclasses) instance, initialized with the system function, encodes a nonlinear equation problem (i.e., a rootfinding problem), and solves it using Newton's method.
#. An instance of a subclass of :py:class:`~skhippr.stability._StabilityMethod._StabilityMethod` provides the functionality to evaluate stability of the problem.
#. The :py:func:`~skhippr.problems.continuation.pseudo_arclength_continuator` wraps a :py:class:`~skhippr.problems.newton.NewtonProblem` into a :py:class:`~skhippr.problems.continuation.BranchPoint` and iteratively predicts, solves and yields the subsequent points on the continuation branch.

The main purpose of SKHiPPR is to provide a framework for comparing various stability analysis methods for periodic solutions and resonance curves in dynamical systems based on the Harmonic Balance method. For this reason, the toolbox is designed to be extensible and modular. For stability determination based on the Hill matrix (Jacobian of the HBM problem), the following classes are available and can be used interchangeably: 

* :py:class:`~skhippr.stability.KoopmanHillProjection.KoopmanHillProjection` (direct Koopman-Hill projection method, cf. Bayer and Leine (2023))
* :py:class:`~skhippr.stability.KoopmanHillProjection.KoopmanHillSubharmonic` (Koopman-Hill projection with subharmonics, cf. Bayer and Leine (2025))
* :py:class:`~skhippr.stability.ClassicalHill.ClassicalHill` (Classical Hill eigenvalue problem with either imaginary-part-based sorting, cf. Colaïtis and Batailly (2022), https://doi.org/10.1016/j.jsv.2022.117219, or symmetry-based sorting, cf. Guillot et al. (2020), https://doi.org/10.1016/j.ijnonlinmec.2024.104894)
* :py:class:`~skhippr.stability.SinglePass.SinglePassRK` (numerical integration using single-pass fixed-step explicit Runge-Kutta methods, cf. Peletan et al. (2013), https://doi.org/10.1007/s11071-012-0744-0)

The modular framework enables easy addition of new problem formulations and comparison of stability methods.

To install SKHiPPR locally on your machine, take a look at the :doc:`installation guide <installation>`. To get started, check out the :doc:`examples` or the :doc:`api`. 

.. A :doc:`full API reference <modules>` is also available, but not recommended for getting started.

Contents
------------
.. toctree::
   :maxdepth: 2

   installation
   examples
   api
   legal_notice

