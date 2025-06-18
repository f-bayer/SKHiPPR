Problem statement
=================

.. contents::

Residual functions
---------------------------

The module :py:mod:`skhippr.systems` provides a zoo of dynamical systems, which are used in the demonstration scripts. However, it is expected that users may want to write their own system functions. 

The following two functions illustrate the syntax requirements for input functions to :py:class:`~skhippr.problems.newton.NewtonProblem` and :py:class:`~skhippr.problems.HBM.HBMProblem`, respectively.

.. autofunction:: skhippr.systems.examples.trivial_newton_residual

.. autofunction:: skhippr.systems.examples.trivial_hbm_system



Solving nonlinear equations
------------------------------------------------------------------------------

.. automodule:: skhippr.problems.newton
    :members: