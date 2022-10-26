
import numpy as np

from metpy.constants import kappa, earth_gravity


def fit_pressure_altitude(pressure, altitude):
    """
    https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.pressure_to_height_std.html
    https://en.wikipedia.org/wiki/Pressure_altitude
    """

    k = 0.190284

    A, B = np.polyfit(np.power(pressure, k), altitude, 1)

    class Converter:

        def p(self, alt):
            return np.power((alt - B) / A, 1 / k)

        def a(self, pre):
            return A * np.power(pre, k) + B

    return Converter()