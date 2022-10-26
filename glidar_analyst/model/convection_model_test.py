import unittest
from convection_model import *


class VelocityIntegration(unittest.TestCase):

    def setUp(self) -> None:
        folder = '../data/Voss-2018-04-29/sounding/'
        sola_file = 'sola_20180331-20180430.nc'
        date = '2018-04-29 11:06:37'

        alt, tmp, pre, dtp = get_observed_data(folder + sola_file, date)

        self.m = ThermalModel(alt, pre, tmp, dtp)

    def test_riemann_integration(self):


        deltaT = 0.5 * units.kelvin
        T_0 = (273.15 + 8) * units.kelvin
        Td_0 = (273.15 + 0) * units.kelvin

        params = ModelParams(T_0, deltaT, Td_0,
                             dew_point_anomaly=0 * units.kelvin,
                             thermal_altitude=100,
                             drag_coeff=0,
                             entrainment_coeff=0,
                             humidity_entrainment_coeff=0,
                             aspect_ratio=0)

        result = ModelResult()

        start = time.time()

        b, _ = self.m.calc_buoyancy_profile(params, result)
        w, z = self.m.calc_velocity_profile(b, self.m.altitude, 300, debug=True)

        # TODO: Hacking units
        b *= units.meter / units.second / units.second

        ww, a = self.m.riemann_calc_velocity_profile(b, self.m.altitude, 300)

        end = time.time()

        plt.plot(ww, a)
        plt.plot(w, z)
        plt.plot(np.interp(z, a, ww), z, 'x')
        plt.show()

        d = np.sum(np.abs(np.interp(z, a, ww) - w)) / w.size

        print(d)
        assert d < 1


class ProfileCalculation(unittest.TestCase):

    def setUp(self) -> None:
        folder = '../data/Voss-2018-04-29/sounding/'
        sola_file = 'sola_20180331-20180430.nc'
        date = '2018-04-29 11:06:37'

        alt, tmp, pre, dtp = get_observed_data(folder + sola_file, date)

        self.m = ThermalModel(alt, pre, tmp, dtp)

        plt.plot(self.m.moist[0], self.m.altitude)
        plt.plot(self.m.moist[-1], self.m.altitude)
        plt.plot(self.m.moist[20], self.m.altitude)
        plt.show()

    def test_cached_profile(self):

        T = 10 * units.celsius
        Td = 0 * units.celsius

        original = self.m.calc_temperature_profile(T, Td)
        new = self.m.get_cached_profile(T, Td)

        print(original)
        print(new)

        plt.plot(new, self.m.altitude)
        plt.plot(original, self.m.altitude)
        plt.show()

        assert np.sum(np.abs(original - new)) < 10 * units.kelvin


if __name__ == '__main__':
    unittest.main()
