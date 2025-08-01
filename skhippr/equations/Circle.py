import numpy as np
from skhippr.equations.AbstractEquation import AbstractEquation


class CircleEquation(AbstractEquation):
    """Has attributes ``y`` and ``radius``.
    The residual function::

        self.y[0] ** 2 + self.y[1] ** 2 - self.radius**2

    is zero when the point ``y`` lies on the circle with radius ``radius``.
    """

    def __init__(self, y: np.ndarray, radius=1):
        super().__init__(None)
        self.y = y
        self.radius = radius

    def residual_function(self):
        return np.atleast_1d(self.y[0] ** 2 + self.y[1] ** 2 - self.radius**2)

    def closed_form_derivative(self, variable):
        match variable:
            case "y":
                return np.atleast_2d(np.array([2 * self.y[0], 2 * self.y[1]]))
            case "radius":
                return np.atleast_2d(-2 * self.radius)
            case _:
                return np.atleast_2d(0)


class AngleEquation(AbstractEquation):
    """Has attributes ``y`` and ``theta``. If ``y`` encloses the angle ``theta`` with the positive x axis, then .
    the residual function::


        self.y[1] * np.cos(self.theta) - self.y[0] * np.sin(self.theta)


    vanishes.
    """

    def __init__(self, y: np.ndarray, theta=1):
        super().__init__(None)
        self.y = y
        self.theta = theta

    def residual_function(self):
        return np.atleast_1d(
            self.y[1] * np.cos(self.theta) - self.y[0] * np.sin(self.theta)
        )

    def closed_form_derivative(self, variable):
        match variable:
            case "y":
                return np.atleast_2d(
                    [
                        -np.sin(np.squeeze(self.theta)),
                        np.cos(np.squeeze(self.theta)),
                    ]
                )
            case "theta":
                return np.atleast_2d(
                    -self.y[1] * np.sin(self.theta) - self.y[0] * np.cos(self.theta)
                )
            case _:
                return np.atleast_2d(0)


class CircleWithPhase(AbstractEquation):
    def __init__(self, x, radius, theta, other):
        super().__init__(stability_method=None)
        self.x = x
        self.radius = radius
        self.theta = theta
        self.other = other

    def residual_function(self):
        return np.squeeze(
            np.array(
                [
                    self.x[0] ** 2 + self.x[1] ** 2 - self.radius**2,
                    self.x[1] * np.cos(self.theta) - self.x[0] * np.sin(self.theta),
                ]
            )
        )

    def closed_form_derivative(self, variable):
        match variable:
            case "x":
                return np.array(
                    [
                        [2 * self.x[0], 2 * self.x[1]],
                        [
                            -np.sin(np.squeeze(self.theta)),
                            np.cos(np.squeeze(self.theta)),
                        ],
                    ]
                )
            case "radius":
                radius = np.atleast_1d(self.radius)
                return np.array([-2 * radius, [0]])
            case "theta":
                return np.array(
                    [
                        [0],
                        -self.x[1] * np.sin(self.theta)
                        - self.x[0] * np.cos(self.theta),
                    ]
                )
            case "other":
                return self.other
            case "other2":
                return self.other[2]
            case _:
                raise NotImplementedError
