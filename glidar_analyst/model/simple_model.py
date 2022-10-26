
import numpy as np

from pint import UnitRegistry
units = UnitRegistry()


class ModelCache:

    def __init__(self, model):

        self.model = model
        self.epsilon = 0.1

        self.cache = {}

    def param_to_vector(self, p):

        return np.array([ p[k] for k in self.model.params ])

    def max_relative_distance(self, p1, p2):

        dp = np.abs(p1 - p2)
        return np.max(dp / np.abs(p1))

    def solve(self, params, epsilon=None):

        result = self.model.solve(params)

        p1 = self.param_to_vector(params)
        self.cache[p1] = result

        return result


class QuadraticModel:

    def __init__(self):

        self.z_max = 2000
        self.N = 50

        self.params = [ 'a', 'b', 'c' ]
        self.names = { p: p.upper() for p in self.params }
        self.short_names = self.names
        self.ranges = {
            'a': (0.00001, 2),
            'b': (-0.00001, -2),
            'c': (1, 2000),
        }
        self.units = {
            'a': units.meter / units.second / units.second,
            'b': units.meter / units.second / units.second,
            'c': units.meter
        }
        self.scale = np.array([ np.max(np.abs(np.array(self.ranges[k]))) for k in self.params ])

        self.R = np.eye(3)

    def map_params(self, params):

        p = np.array(list(params[k] for k in self.params))
        p = p / self.scale
        p = np.matmul(self.R, p)
        return dict( zip(self.params, p * self.scale) )

    def map_inverse_params(self, p):

        p = np.array(list(p[k] for k in self.params))
        p = p / self.scale
        p = np.matmul(self.R.T, p)
        return dict( zip(self.params, p * self.scale) )

    def solve(self, params):

        params = self.map_params(params)

        a = np.abs(params['a'])
        b = params['b']
        c = params['c']

        if a == 0 or b == 0:
            raise RuntimeError('Unsupported parameter value', a, b)

        za = np.linspace(0, c, self.N)
        zb = np.linspace(c, self.z_max, self.N)

        wa = np.sqrt(np.abs(2 * a * za))

        wb2 = 2 * b * (zb - c * (1 - a / b))
        wb2[wb2 < 0] = 0
        wb = np.sqrt(wb2)

        return np.concatenate([za, zb]), np.concatenate([wa, wb])


###########################################################################
# junk code below
if __name__ == '__main__':
    
    dT = 0.3
    T = 8 + 273.15
    a = 0.0023 # drag
    B = 0.000161 # extinction through mixing
    z_max = 2000
    dz = 0.1 # m


    def simple_thermal(dT: float, T: float, a: float, B: float, z_max: float, dz: float):

        b_0 = dT / T * 9.81

        z = np.linspace(0, z_max, int(z_max / dz))

        b = b_0 * np.exp(-B*z)

        dt = 0.1
        N = 1000000

        w = np.zeros(N)
        zz = np.zeros(N)

        for i in range(N-1):
            if zz[i] > z_max:
                break
            w[i+1] = w[i] + (-a * w[i] + b[int(zz[i] / dz)] ) * dt
            zz[i+1] = zz[i] + w[i+1] * dt

        return w[:i], zz[:i]

