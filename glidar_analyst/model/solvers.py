

import numpy as np
from numpy.linalg import LinAlgError

from pint import UnitRegistry
units = UnitRegistry()


class NewtonSolver:

    def __init__(self, model):

        self.model = model
        self.epsilon = 0.01

    def get_derivative(self, params, variable, result):

        params = params.copy()
        # r1 = self.model.solve(params)               # optimize out later
        r1 = result

        a, b = self.model.ranges[variable]
        dv = self.epsilon * (b-a)
        params[variable] += dv
        r2 = self.model.solve(params)

        r = np.interp(r1[0], r2[0], r2[1])
        return r1[0], (r - r1[1]) / dv

    def get_derivative_at_x(self, params, variable, result, target_x):

        d = self.get_derivative(params, variable, result)
        dy_dv = np.interp(target_x, *d)

        return dy_dv

    def solve_for(self, target_x, target_y, variable, params, max_it=10):

        print(max_it, params)
        if max_it < 0:
            raise RuntimeError('Cannot find solution in time.', params, variable)

        params = params.copy()
        r = self.model.solve(params)
        y = np.interp(target_x, *r)
        if np.abs((y - target_y) / target_y) < self.epsilon:
            return params

        d = self.get_derivative(params, variable, r)
        dy_dv = np.interp(target_x, *d)
        dv = (target_y - y)
        if dy_dv != 0:
            dv /= dy_dv
        else:
            dv *= 10
        params[variable] += dv

        return self.solve_for(target_x, target_y, variable, params, max_it=max_it-1)

    def solve_multi_target(self, params, targets, max_it=10):

        if len(targets) < 1:
            return params

        print(max_it, params)
        if max_it < 0:
            raise RuntimeError('Cannot find solution in time.', params)

        ranges = self.model.ranges

        r = self.model.solve(params)
        n = len(targets)
        M = np.zeros((n,n))
        Y = np.zeros((n,1))
        Y_frac = np.zeros((n,1))

        for i, vi in enumerate(targets.values()):
            y = np.interp(vi.target_x, *r)
            Y[i,0] = vi.target_y - y
            Y_frac[i,0] = Y[i,0] / vi.target_y

        if np.max(np.abs(Y_frac)) < self.epsilon:
            return params

        for i, vi in enumerate(targets.values()):
            for j, vj in enumerate(targets.values()):
                M[i,j] = self.get_derivative_at_x(params, vj.variable, r, vi.target_x)

        # try:
        X = np.linalg.solve(M, Y)
        # except LinAlgError as lae:
        #     print(lae)
        #     return params

        p = params.copy()
        for i, v in enumerate(targets.values()):
            p[v.variable] += X[i,0]
            if p[v.variable] > ranges[v.variable][1]:
                p[v.variable] = ranges[v.variable][1]
            if p[v.variable] < ranges[v.variable][0]:
                p[v.variable] = ranges[v.variable][0]

        return self.solve_multi_target(p, targets, max_it-1)


class BisectionSolver:

    def __init__(self, model):

        self.model = model
        self.epsilon = 0.01     # one percent precision

    def solve_for(self, target_x, target_y, variable, params):
        """
        Tries to find parameters such that the model result is close to
        the specified target x and y values.

        :param target_x: Desired value of x
        :param target_y: Desired value of y
        :param variable: Variable to optimize
        :param params: initial guess, fixing all other params at this value
        :return: The found parameters solving x and y
        """

        # Get left and right x values
        t0, t1 = self.model.ranges[variable]

        # Solve left side
        params[variable] = t0
        x, y = self.model.solve(params)
        w_left = np.interp(target_x, x, y)

        params[variable] = t1
        x, y = self.model.solve(params)
        w_right = np.interp(target_x, x, y)

        if (w_left - target_y) * (w_right - target_y) > 0:
            raise RuntimeError('No solution in the given interval', t0, t1)

        for i in range(100):
            print(t0, t1, w_left, w_right)
            params[variable] = 0.5*t1 + 0.5*t0
            x, y = self.model.solve(params)
            w_mid = np.interp(target_x, x, y)

            if (w_mid - target_y) * (w_right - target_y) < 0:
                w_left = w_mid
                t0 = 0.5 * t1 + 0.5 * t0
            else:
                w_right = w_mid
                t1 = 0.5 * t1 + 0.5 * t0

            if np.abs((w_mid - target_y) / target_y) < 0.01:
                break

        return params
