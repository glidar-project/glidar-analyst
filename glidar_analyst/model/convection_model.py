import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import metpy.calc as mpcalc
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units
from scipy.integrate import BDF
from scipy import interpolate
from scipy.ndimage.filters import maximum_filter1d, gaussian_filter1d

import threading

import os
import netCDF4 as nc
import xarray as xr
import time


class Utils:

    @staticmethod
    def combine_pint_arrays(a, b, idx):

        if not a.check(b.units):
            raise ValueError('Incompatible units', a, b)

        b_star = b.to(a.units)
        A = a.magnitude
        B = b_star.magnitude
        A[idx:] = B[idx:]
        return A * a.units


class AirProfile:
    """
    This is the class containing sounding data
    """

    def __init__(self, altitude, pressure, temperature, dewpoint):
        """
        The constructor taking the observation data

        :param altitude: the altitude
        :param pressure: the pressure
        :param temperature: air temperature
        """
        self.altitude = altitude
        self.pressure = pressure
        self.temperature = temperature
        self.dewpoint = dewpoint


class ModelParams:
    """
    The surface initial conditions. The units need to be included.
    """

    def __init__(
        self,
        surface_temperature,
        temperature_anomaly,
        dew_point_temperature,
        dew_point_anomaly,
        thermal_altitude,
        drag_coeff,
        entrainment_coeff,
        humidity_entrainment_coeff,
        aspect_ratio=0,
        quadratic_drag_coeff=0,
    ):
        self.surface_temperature = surface_temperature
        self.temperature_anomaly = temperature_anomaly
        self.dew_point_temperature = dew_point_temperature
        self.dew_point_anomaly = dew_point_anomaly
        self.thermal_altitude = thermal_altitude
        self.drag_coeff = drag_coeff
        self.quadratic_drag_coeff = quadratic_drag_coeff
        self.entrainment_coeff = entrainment_coeff
        self.humidity_entrainment_coeff = humidity_entrainment_coeff
        self.aspect_ratio = aspect_ratio


class ModelResult:
    """
    This is the class containing the result of a model run.

    All the shite should be including units.
    """

    def __init__(self):

        # Params that made this result
        self.params = None
        self.air_profile = None

        # Vertical velocity profiles
        self.velocity_profile = None
        self.virtual_velocity_profile = None

        # Corresponding buoyancy profiles
        self.buoyancy = None
        self.virtual_buoyancy = None
        self.effective_buoyancy = None
        self.effective_virtual_buoyancy = None

        # Potential temperatures -- not used for vis
        self.theta_bar = None
        self.theta_bar_virtual = None
        self.theta_virtual = None
        self.virtual_delta_t = None

        # Absolute temperatures
        # t_pert = t_bar + t_prime
        self.t_bar = None
        self.t_bar_virtual = None

        self.t_pert = None
        self.t_pert_virtual = None
        self.t_ent = None
        self.t_ent_virtual = None

        self.dew_bar = None
        self.dew_pert = None
        self.dew_ent = None

        self.base_mixing_line = None
        self.pert_mixing_line = None

        self.lcl_pressure = None
        self.lcl_temperature = None

        self.added_water = None
        self.cloud_water = None


class ThermalModel:
    """
    This is the mighty class computing vertical profiles from air sounding data.
    """
    semaphore = threading.BoundedSemaphore()
    g = 9.81  # the gravitational constant ms^-2

    def __init__(self,
                 altitude,
                 pressure,
                 temperature,
                 dewpoint,
                 cached=True,
                 riemann=True):
        """
        The constructor taking the observation data

        :param altitude: the altitude of the observations
        :param pressure: the pressure
        :param temperature: air temperature
        """
        self.cached = cached
        self.riemann_flag = riemann

        p = np.linspace(pressure[0], pressure[-1], pressure.size)
        self.altitude = np.interp(p[::-1], pressure[::-1],
                                  altitude[::-1])[::-1]
        self.temperature = np.interp(p[::-1], pressure[::-1],
                                     temperature[::-1])[::-1]
        self.dewpoint = np.interp(p[::-1], pressure[::-1],
                                  dewpoint[::-1])[::-1]
        self.pressure = p

        self.air_profile = AirProfile(self.altitude, self.pressure,
                                      self.temperature, self.dewpoint)

        self.moist = None
        if self.cached:
            self.precompute_cache()

    def precompute_cache(self):

        self.moist = []
        # Precompute the moist adiabatic profiles
        temp = np.linspace(-20, 30, 50)
        for t in temp:
            m = mpcalc.moist_lapse(self.pressure, t * units.celsius)
            self.moist.append(m)
        self.moist = np.array(self.moist) * units.celsius
        print('Cache computed.')

    def get_cached_profile(self, temperature, dew_point):

        if self.moist is None:
            self.precompute_cache()
            print('Cache was not precomputed, computing now...')

        p0, t0 = mpcalc.lcl(self.pressure[0], temperature, dew_point)
        dry = mpcalc.dry_lapse(self.pressure, temperature)

        ip = None
        ips = np.where(self.pressure < p0)
        if len(ips[0]) > 0:
            ip = ips[0][0]
        else:
            return dry
            # raise RuntimeError('Cannot get cached profile, pressure too small', p0, t0, temperature, dew_point)

        it = None
        its = np.where(self.moist.T[ip] > t0)
        if len(its[0] > 0):
            it = its[0][0]
            if it == 0:
                raise RuntimeError(
                    'Cannot get cached profile, temperature too low.', p0, t0,
                    temperature, dew_point)
            it = (it - 1, it)
        else:
            raise RuntimeError(
                'Cannot get cached profile, temperature too high.', p0, t0,
                temperature, dew_point)

        # Bi-linear interpolation
        pp = self.pressure[[ip - 1, ip]]
        ttt = self.moist[it[0]:it[1] + 1, [ip - 1, ip]]
        tt = (ttt[:, 0] * (pp[1] - p0) +
              (p0 - pp[0]) * ttt[:, 1]) / (pp[1] - pp[0])

        f = ((tt[1] - t0) * self.moist[it[0], :] +
             (t0 - tt[0]) * self.moist[it[1], :]) / (tt[1] - tt[0])

        return np.where(self.pressure > p0, dry, f)

    def calc_temperature_profile(self, temperature, dew_point):
        """
        Helper function calling MetPy to calculate temperature profiles

        :param temperature: surface temperature
        :param dew_point: surface dew point
        :return: temperature profile
        """
        if self.cached:
            return self.get_cached_profile(temperature, dew_point)

        with ThermalModel.semaphore:
            res = mpcalc.parcel_profile(self.pressure, temperature,
                                        dew_point).to(units.kelvin)

        return res

    @staticmethod
    def exp_decay(t_prime, z, gamma):
        """
        Approximates the entrainment of the temperature profile.

        """
        # z = self.altitude.magnitude

        dz = z[1:] - z[:-1]
        dt_prime = t_prime[1:] - t_prime[:-1]
        t = np.zeros_like(t_prime)
        t[0] = t_prime[0]
        for i, (dt, dz) in enumerate(zip(dt_prime, dz)):
            t[i + 1] = t[i] + dt - dz * gamma * t[i]
        return t

    @staticmethod
    def calculate_RH_profile(pre, tmp, rh_surface):
        """
        Calculates the relative humidity profile.
        It computes the mixing ratio at the surface level and 
        then extrapolateds the values for RH using a constant 
        mixing ratio. After condensation the RH is capped.
        """
        mx_rel_surface = mpcalc.mixing_ratio_from_relative_humidity(
            pre[0], tmp[0], rh_surface)
        RH = mpcalc.relative_humidity_from_mixing_ratio(
            pre, tmp, mx_rel_surface)
        RH[RH > 1] = 1

        return RH

    @staticmethod
    def decay_MX_profile(mx_base, mx_prime, alt, gamma):
        """
        Computes the exponential mixing for the mixing ratio.
        """
        dmx = mx_prime - mx_base
        delta = ThermalModel.exp_decay(dmx.magnitude, alt, gamma) * dmx.units

        return mx_base + delta

    @staticmethod
    def decay_RH_profile(tmp, RH_base, RH_prime, alt, gamma):
        """
        Computes the exponential mixing for the RH.
        The RH is transformed into the dew point and the
        exponential decay is applide to the dew point anomaly.
        The decision to compute it in the dew point is arbit-
        rary and not checked against anything. The specific humidity 
        might be a better choice.
        """
        dp_base = mpcalc.dewpoint_from_relative_humidity(tmp, RH_base)
        dp_prime = mpcalc.dewpoint_from_relative_humidity(tmp, RH_prime)
        ddp = dp_prime - dp_base
        delta = ThermalModel.exp_decay(ddp.magnitude, alt, gamma) * ddp.units

        res = mpcalc.relative_humidity_from_dewpoint(tmp, dp_base + delta)
        return res

    @staticmethod
    def calculate_virtual_profile(pressure, temperature, RH_profile):
        """
        Computes the virtual *potential* temperature
        
        @dew_point surface dew point
        """
        # mixing ratio based on the relative humidity
        modelled_mx = mpcalc.mixing_ratio_from_relative_humidity(
            pressure, temperature, RH_profile)
        # Finally the virtual temprarture
        modelled_virtual_temperature = mpcalc.virtual_potential_temperature(
            pressure, temperature, modelled_mx)

        return modelled_virtual_temperature

    @staticmethod
    def entrain_variable(background, parcel, altitude, coeff):

        ddp = parcel - background
        delta = ThermalModel.exp_decay(ddp.magnitude, altitude.magnitude,
                                       coeff) * ddp.units

        return background + delta

    @staticmethod
    def calculate_unreal_RH_profile(pre, tmp, rh_surface):
        """
        Calculates the relative humidity profile.
        It computes the mixing ratio at the surface level and 
        then extrapolateds the values for RH using a constant 
        mixing ratio. After condensation the RH is *NOT* capped.
        """
        mx_rel_surface = mpcalc.mixing_ratio_from_relative_humidity(
            pre[0], tmp[0], rh_surface)
        RH = mpcalc.relative_humidity_from_mixing_ratio(
            pre, tmp, mx_rel_surface)

        return RH

    @staticmethod
    def calculate_entrained_temperature_and_dewpoint(pressure, altitude, t_bar,
                                                     dew_bar, surf_temp,
                                                     surf_dewpoint, t_ent,
                                                     d_ent):

        dry = mpcalc.dry_lapse(pressure, surf_temp)

        dry_ent = ThermalModel.entrain_variable(t_bar, dry, altitude, t_ent)

        rh = ThermalModel.calculate_unreal_RH_profile(
            pressure, dry_ent,
            mpcalc.relative_humidity_from_dewpoint(surf_temp, surf_dewpoint))

        dew = mpcalc.dewpoint_from_relative_humidity(dry_ent, rh)

        dew_ent = ThermalModel.entrain_variable(dew_bar, dew, altitude, d_ent)

        # Intersecting profiles
        result_t = np.zeros_like(dry_ent) * dry_ent.units
        result_d = np.zeros_like(dry_ent) * dry_ent.units

        m = dew_ent < dry_ent
        idx = np.argmax(~m)

        if dry_ent[idx] > dew_ent[idx]:

            return dry_ent, dew_ent, None, None

        if idx > 0:
            result_t[:idx] = dry_ent[:idx]
            result_d[:idx] = dew_ent[:idx]

        moist = mpcalc.moist_lapse(pressure[idx:], dew_ent[idx])
        # moist_ent = ThermalModel.entrain_variable(t_bar[idx:], moist, altitude[idx:], t_ent)

        result_t[idx:] = moist
        result_d[idx:] = moist

        lcl_pre = pressure[idx]
        lcl_temp = result_t[idx]

        return result_t, result_d, lcl_pre, lcl_temp

    def calc_virtual_buoyancy_profile_with_custom_ent(self,
                                                      mp: ModelParams,
                                                      result: ModelResult,
                                                      use_sounding=True):

        pre = self.pressure
        tmp = self.temperature

        # Calculate the base state parcel profile called t_bar
        t_bar = self.calc_temperature_profile(mp.surface_temperature,
                                              mp.dew_point_temperature)
        if use_sounding:
            idx = np.where((t_bar < tmp) & (self.altitude.magnitude > mp.thermal_altitude))[0]
            if idx.size > 0:
                idx = idx[0]
                t_bar = Utils.combine_pint_arrays(t_bar, tmp, idx)
            else:
                idx = None
        result.t_bar = t_bar

        # Calculate the base RH profile and the corresponding virtual potential temperature
        RH_bar = self.calculate_RH_profile(
            pre, t_bar,
            mpcalc.relative_humidity_from_dewpoint(mp.surface_temperature,
                                                   mp.dew_point_temperature))
        if use_sounding:
            rh = mpcalc.relative_humidity_from_dewpoint(tmp, self.dewpoint)
            if idx is not None:
                # RH_bar[idx:] = rh[idx:]
                RH_bar = Utils.combine_pint_arrays(RH_bar, rh, idx)

            mx_obs = mpcalc.mixing_ratio_from_relative_humidity(pre, tmp, rh)
            mx_bar = mpcalc.mixing_ratio_from_relative_humidity(
                pre, t_bar, RH_bar)

            result.base_mixing_line = mx_bar

            water_obs = mpcalc.density(pre, tmp, mx_obs) * mx_obs
            water_bar = mpcalc.density(pre, t_bar, mx_bar) * mx_bar

            result.added_water = np.trapz(water_bar - water_obs, self.altitude)

        theta_bar_virtual = self.calculate_virtual_profile(pre, t_bar, RH_bar)
        result.theta_bar_virtual = theta_bar_virtual
        result.t_bar_virtual = (
            theta_bar_virtual / mpcalc.potential_temperature(
                self.pressure,
                np.ones_like(self.pressure) * units.kelvin)) * units.kelvin
        result.dew_bar = mpcalc.dewpoint_from_relative_humidity(t_bar, RH_bar)

        # Calculate the entrained perturbed state 
        t_ent, dew_ent, lcl_pre, lcl_temp = self.calculate_entrained_temperature_and_dewpoint(
            pre,
            self.altitude,
            t_bar,
            result.dew_bar,
            surf_temp=mp.surface_temperature + mp.temperature_anomaly,
            surf_dewpoint=mp.dew_point_temperature + mp.dew_point_anomaly,
            t_ent=mp.entrainment_coeff,
            d_ent=mp.humidity_entrainment_coeff)
        
        result.t_ent = t_ent
        result.dew_ent = dew_ent
        result.lcl_pressure = lcl_pre
        result.lcl_temperature = lcl_temp

        # Calculate the basic perturbed state
        t = self.calc_temperature_profile(
            mp.surface_temperature + mp.temperature_anomaly,
            mp.dew_point_temperature + mp.dew_point_anomaly)
        RH = self.calculate_RH_profile(
            pre, t,
            mpcalc.relative_humidity_from_dewpoint(
                mp.surface_temperature + mp.temperature_anomaly,
                mp.dew_point_temperature + mp.dew_point_anomaly))
        result.t_pert = t
        result.dew_pert = mpcalc.dewpoint_from_relative_humidity(t, RH)
        theta_pert_virtual = self.calculate_virtual_profile(pre, t, RH)

        # Calculate virtual profiles
        RH = mpcalc.relative_humidity_from_dewpoint(t_ent, dew_ent)
        theta_virtual = self.calculate_virtual_profile(pre, t_ent, RH)
        result.theta_virtual = theta_virtual

        theta_prime_virtual = theta_virtual - theta_bar_virtual

        # The buoyancy computed directly from temperature perturbation
        b = ThermalModel.g * theta_prime_virtual / theta_bar_virtual
        result.virtual_buoyancy = b

        theta_one = 1 * units.kelvin / mpcalc.potential_temperature(
            self.pressure,
            np.ones_like(self.pressure) * units.kelvin)

        result.t_pert_virtual = theta_pert_virtual * theta_one

        T_ent = theta_virtual * theta_one
        result.t_ent_virtual = T_ent

        return b, T_ent

    def calc_virtual_buoyancy_profile(self,
                                      mp: ModelParams,
                                      result: ModelResult,
                                      use_sounding=True):

        pre = self.pressure
        tmp = self.temperature

        # Calculate the base state parcel profile called t_bar
        t_bar = self.calc_temperature_profile(mp.surface_temperature,
                                              mp.dew_point_temperature)
        if use_sounding:
            idx = np.where(t_bar < tmp)[0]
            if idx.size > 0:
                idx = idx[0]
                t_bar = Utils.combine_pint_arrays(t_bar, tmp, idx)
            else:
                idx = None
        result.t_bar = t_bar

        # Calculate the base RH profile and the corresponding virtual potential temperature
        RH_bar = self.calculate_RH_profile(
            pre, t_bar,
            mpcalc.relative_humidity_from_dewpoint(mp.surface_temperature,
                                                   mp.dew_point_temperature))
        if use_sounding:
            rh = mpcalc.relative_humidity_from_dewpoint(tmp, self.dewpoint)
            if idx is not None:
                # RH_bar[idx:] = rh[idx:]
                RH_bar = Utils.combine_pint_arrays(RH_bar, rh, idx)

            mx_obs = mpcalc.mixing_ratio_from_relative_humidity(pre, tmp, rh)
            mx_bar = mpcalc.mixing_ratio_from_relative_humidity(
                pre, t_bar, RH_bar)

            result.base_mixing_line = mx_bar

            water_obs = mpcalc.density(pre, tmp, mx_obs) * mx_obs
            water_bar = mpcalc.density(pre, t_bar, mx_bar) * mx_bar

            result.added_water = np.trapz(water_bar - water_obs, self.altitude)

        theta_bar_virtual = self.calculate_virtual_profile(pre, t_bar, RH_bar)
        result.theta_bar_virtual = theta_bar_virtual
        result.t_bar_virtual = (
            theta_bar_virtual / mpcalc.potential_temperature(
                self.pressure,
                np.ones_like(self.pressure) * units.kelvin)) * units.kelvin
        result.dew_bar = mpcalc.dewpoint_from_relative_humidity(t_bar, RH_bar)

        # Calculate the perturbed state
        t = self.calc_temperature_profile(
            mp.surface_temperature + mp.temperature_anomaly,
            mp.dew_point_temperature + mp.dew_point_anomaly)
        # result.t_pert_virtual = t
        RH = self.calculate_RH_profile(
            pre, t,
            mpcalc.relative_humidity_from_dewpoint(
                mp.surface_temperature + mp.temperature_anomaly,
                mp.dew_point_temperature + mp.dew_point_anomaly))
        result.dew_pert = mpcalc.dewpoint_from_relative_humidity(t, RH)

        # decay the RH profile
        mx = self.decay_MX_profile(
            mpcalc.mixing_ratio_from_relative_humidity(pre, t_bar, RH_bar),
            mpcalc.mixing_ratio_from_relative_humidity(pre, t, RH),
            self.altitude.magnitude,
            gamma=mp.humidity_entrainment_coeff)
        result.pert_mixing_line = mx

        RH = mpcalc.relative_humidity_from_mixing_ratio(pre, t, mx)

        theta_virtual = self.calculate_virtual_profile(pre, t, RH)
        result.theta_virtual = theta_virtual
        result.dew_ent = mpcalc.dewpoint_from_relative_humidity(t, RH)

        theta_prime_virtual = theta_virtual - theta_bar_virtual

        # The diffusion
        z = self.altitude.magnitude
        theta_prime_virtual = self.exp_decay(
            theta_prime_virtual.magnitude, z,
            mp.entrainment_coeff) * theta_prime_virtual.units

        # The buoyancy computed directly from temperature perturbation
        b = ThermalModel.g * theta_prime_virtual / theta_bar_virtual
        result.virtual_buoyancy = b

        theta_ent = theta_bar_virtual + theta_prime_virtual

        theta_one = 1 * units.kelvin / mpcalc.potential_temperature(
            self.pressure,
            np.ones_like(self.pressure) * units.kelvin)

        T_ent = theta_ent * theta_one
        result.t_ent_virtual = T_ent

        result.t_pert_virtual = result.theta_virtual * theta_one
        return b, T_ent

    def calc_buoyancy_profile(self,
                              mp: ModelParams,
                              result: ModelResult,
                              use_sounding=True):

        surface_temperature = mp.surface_temperature
        dew_point = mp.dew_point_temperature
        temperature_anomaly = mp.temperature_anomaly
        gamma = mp.entrainment_coeff

        # Calculate the base state parcel profile called t_bar
        t_bar = self.calc_temperature_profile(surface_temperature, dew_point)

        if use_sounding:
            idx = np.where(t_bar < self.temperature)[0]
            if idx.size > 0:
                idx = idx[0]
                # t_bar[idx:] = tmp[idx:]
                t_bar = Utils.combine_pint_arrays(t_bar, self.temperature, idx)
            else:
                idx = None

        # if use_sounding:
        # If we are using a sounding profile, combine the base state with
        # the sounding temperatures
        # t_bar = np.where(t_bar < self.temperature, self.temperature, t_bar)

        result.t_bar = t_bar

        # Theta is the potential temperature computed from the absolute temperature
        # Theta bar is then potential temperature of the base state
        theta_bar = mpcalc.potential_temperature(self.pressure, t_bar)
        result.theta_bar = theta_bar

        # This is the potential temperature of the perturbed state
        # This is what gives the thermal any energy
        theta = mpcalc.potential_temperature(
            self.pressure,
            self.calc_temperature_profile(
                surface_temperature + temperature_anomaly, dew_point))

        # The perturbation explicitely
        theta_prime = theta - theta_bar

        # The diffusion
        z = self.altitude.magnitude
        theta_prime = self.exp_decay(theta_prime.magnitude, z,
                                     gamma) * theta_prime.units

        # The buoyancy computed directly from temperature perturbation
        b = ThermalModel.g * theta_prime / theta_bar
        result.buoyancy = b

        theta_ent = theta_bar + theta_prime
        T_ent = theta_ent / mpcalc.potential_temperature(
            self.pressure,
            np.ones_like(self.pressure) * units.kelvin)

        T_ent = T_ent * units.kelvin
        result.t_ent = T_ent
        return b, T_ent

    def calculate_back_pressure_term(self, aspect_ratio):
        """
        Calculates the effective buoyancy according to Jeevanjee and Romps [1],
        assuming a cylindrical thermal with aspect ratio D/H.
        [1] https://doi.org/10.1002/qj.2683

        :param aspect_ratio: width devided by height of the thermal
        :return: effective buoyancy
        """
        effective_buoyancy = 1 / np.sqrt(1 + aspect_ratio**2)
        return effective_buoyancy

    @staticmethod
    def riemann_calc_velocity_profile(buoyancy,
                                      altitude,
                                      z_0,
                                      alpha=0.00,
                                      quadratic_drag=0,
                                      w0=0.01):
        """
        Found a way how to simplify the integration

        :param buoyancy: with units
        :param altitude: with units
        :param z_0:
        :param alpha:
        :param debug:
        :param w0:
        :return:
        """
        buoyancy = buoyancy.magnitude
        altitude = altitude.magnitude
        # alpha = alpha / units.meter
        dz = (altitude[1:] - altitude[:-1])  # * units.meter
        w2 = np.zeros_like(
            altitude
        )  # * units.meter * units.meter / units.second / units.second

        # alpha *= units.meter / units.second
        # quadratic_drag /= units.meter

        i0 = 0
        iii = np.where(altitude > z_0)[0]
        if len(iii):
            i0 = iii[0]

        for i, dz in enumerate(dz):

            # skip the values below the starting altitude
            if i < i0:
                continue

            # Integrate the buoyancy contributions
            w = w2[i] + (buoyancy[i] - alpha * np.sqrt(w2[i]) -
                         quadratic_drag * w2[i]) * 2 * dz

            # Break if vertical velocity becomes negative
            if w < 0:
                break
            w2[i + 1] = w

        return np.sqrt(w2[i0:i + 1]), altitude[i0:i + 1]

    @staticmethod
    def calc_velocity_profile_odeint(buoyancy,
                                     altitude,
                                     z_0,
                                     alpha=0.00,
                                     quadratic_drag=0,
                                     debug=False):
        """
        Integrates the buoyancy profile to obtain the velocity
        :param buoyancy:
        :param altitude:
        :param z_0:
        :param alpha:
        :return: velocity, altitude
        """

        class B:

            def __init__(self, a, b):
                """
                @param a: altitude
                @param b: buoyancy
                """
                self.x = a
                self.y = b
                self.tck = interpolate.splrep(self.x, self.y, s=0)

            def b(self, z):
                return interpolate.splev(z, self.tck, der=0)

            def f(self, t, y):  # time is z, and y is w
                return self.b(t) / y - alpha * y - quadratic_drag * y * y

        bb = B(altitude.magnitude, buoyancy.magnitude)
        r = BDF(bb.f, t0=z_0, y0=[0.01], t_bound=3000)

        res = []
        for i in range(1000):
            try:
                res.append((r.t, r.y))
                r.step()
            except RuntimeError:
                # print('Took me {} steps'.format(i))
                break
        res = np.array(res, dtype=float)

        if debug:
            return res[:, 1], res[:, 0], bb

        return res[:, 1], res[:, 0]

    def calc_velocity_profile(self,
                              buoyancy,
                              altitude,
                              z_0,
                              alpha=0.00,
                              quadratic_drag=0,
                              debug=False):

        if self.riemann_flag:
            return self.riemann_calc_velocity_profile(buoyancy, altitude, z_0,
                                                      alpha, quadratic_drag)
        else:
            return self.calc_velocity_profile_odeint(buoyancy, altitude, z_0,
                                                     alpha, quadratic_drag)

    def compute_only_humid_fit(self, model_params: ModelParams):

        result = ModelResult()
        result.params = model_params
        result.air_profile = self.air_profile

        a = self.calculate_back_pressure_term(model_params.aspect_ratio)

        # Calculate the humid model using virtual temperature
        # b, T_ent = self.calc_virtual_buoyancy_profile(model_params, result)
        b, T_ent = self.calc_virtual_buoyancy_profile_with_custom_ent(model_params, result)
        result.effective_virtual_buoyancy = b * a
        try:
            w, z = self.calc_velocity_profile(
                b * a,
                self.altitude,
                model_params.thermal_altitude,
                alpha=model_params.drag_coeff * a,
                quadratic_drag=model_params.quadratic_drag_coeff * a)
            result.virtual_velocity_profile = (w, z)

        except ValueError as ve:
            print(ve)

        return result

    def compute_fit(self, model_params: ModelParams):

        result = ModelResult()
        result.params = model_params
        result.air_profile = self.air_profile
        result.lcl_pressure, result.lcl_temperature = mpcalc.lcl(
            self.pressure[0], model_params.surface_temperature +
            model_params.temperature_anomaly,
            model_params.dew_point_temperature +
            model_params.dew_point_anomaly)

        # Calculate the dry model vertical velocity
        b, T_ent = self.calc_buoyancy_profile(model_params, result)
        a = self.calculate_back_pressure_term(model_params.aspect_ratio)
        result.effective_buoyancy = b * a
        z = None
        try:
            w, z = self.calc_velocity_profile(
                b * a,
                self.altitude,
                model_params.thermal_altitude,
                alpha=model_params.drag_coeff * a,
                quadratic_drag=model_params.quadratic_drag_coeff * a)
            result.velocity_profile = (w, z)
        except ValueError as ve:
            print(ve)

        # Calculate the humid model using virtual temperature
        # b, T_ent = self.calc_virtual_buoyancy_profile(model_params, result)
        b, T_ent = self.calc_virtual_buoyancy_profile_with_custom_ent(model_params, result)
        result.effective_virtual_buoyancy = b * a
        try:
            w, z = self.calc_velocity_profile(
                b * a,
                self.altitude,
                model_params.thermal_altitude,
                alpha=model_params.drag_coeff * a,
                quadratic_drag=model_params.quadratic_drag_coeff * a)
            result.virtual_velocity_profile = (w, z)

        except ValueError as ve:
            print(ve)

        # Fuck cloud water for now
        # if z is not None and z.size > 0:
        #     w_prime = mpcalc.density(self.pressure, result.t_pert, result.pert_mixing_line) * result.pert_mixing_line
        #     w_base = mpcalc.density(self.pressure, result.t_bar, result.base_mixing_line) * result.base_mixing_line
        #
        #     w = w_prime - w_base
        #     z *= units.meter
        #     water = np.trapz(w[self.altitude<z[-1]], self.altitude[self.altitude<z[-1]])
        #     result.cloud_water = water

        # Calculate the reference delta T between the dry and humid model
        virtual_theta_prime = (result.virtual_buoyancy - result.buoyancy
                               ) * result.theta_bar / ThermalModel.g
        virtual_t_prime = virtual_theta_prime / mpcalc.potential_temperature(
            self.pressure,
            np.ones_like(self.pressure) * units.kelvin)
        result.virtual_delta_t = virtual_t_prime * units.kelvin
        return result

    def fit_thermal(self, df, x0, bounds):

        w, z = df['vario'].to_numpy(), df['altitude'].to_numpy()

        zz = np.linspace(0, 2000, 200)
        ww = np.zeros_like(zz)
        idx = (z / 10.).astype(np.int)

        for i, j in enumerate(idx):
            if ww[j] < w[i]:
                ww[j] = w[i]

        sub = np.nonzero(ww)

        def fit_fn(args):
            T_0 = args[0]
            Td_0 = args[1]
            deltaT = args[2]
            a = args[3]
            g = args[4]
            z0 = args[5]
            b, _ = self.calc_buoyancy_profile(T_0 * units.kelvin,
                                              Td_0 * units.kelvin,
                                              deltaT * units.kelvin,
                                              gamma=g)
            w, z = self.calc_velocity_profile(b, self.altitude, z0, alpha=a)

            return np.sum((ww - np.interp(zz, z, w))**2)

        from scipy.optimize import minimize

        return ww, zz, minimize(fit_fn,
                                x0,
                                method='Nelder-Mead',
                                bounds=bounds)


###################################################################################################################
# Some testing functionality below ...
###################################################################################################################


def test_riemann_integration():

    folder = '../data/Voss-2018-04-29/sounding/'
    sola_file = 'sola_20180331-20180430.nc'
    date = '2018-04-29 11:06:37'

    alt, tmp, pre, dtp = get_observed_data(folder + sola_file, date)

    deltaT = 0.5 * units.kelvin
    T_0 = (273.15 + 8) * units.kelvin
    Td_0 = (273.15 + 0) * units.kelvin

    params = ModelParams(T_0,
                         deltaT,
                         Td_0,
                         dew_point_anomaly=0 * units.kelvin,
                         thermal_altitude=100,
                         drag_coeff=0,
                         entrainment_coeff=0,
                         humidity_entrainment_coeff=0,
                         aspect_ratio=0)

    result = ModelResult()
    start = time.time()

    m = ThermalModel(alt, pre, tmp, dtp)
    b, _ = m.calc_buoyancy_profile(params, result)
    w, z, bb = m.calc_velocity_profile(b, alt, 300, debug=True)
    # TODO: Hacking units
    b *= units.meter / units.second / units.second

    for w0 in np.logspace(-1, -5, 10):
        # print(w0)
        ww, a = m.riemann_calc_velocity_profile(b, alt.values, 300, w0=w0)
        plt.plot(ww, a)

    end = time.time()

    # print(ww)

    # print('Timed average over 100 runs:', (end - start) / 100)

    plt.plot(w, z, 'x')
    plt.show()


def time_model_computation():

    from tqdm.cli import tqdm

    alt, tmp, pre, dtp = get_observed_data()

    params = ModelParams(surface_temperature=(273.15 + 8) * units.kelvin,
                         temperature_anomaly=0.5 * units.kelvin,
                         dew_point_temperature=(273.15 + 0) * units.kelvin,
                         dew_point_anomaly=0 * units.kelvin,
                         thermal_altitude=100,
                         drag_coeff=0,
                         entrainment_coeff=0,
                         humidity_entrainment_coeff=0,
                         aspect_ratio=0)

    for flags in [(False, False), (True, False), (False, True), (True, True)]:
        m = ThermalModel(alt, pre, tmp, dtp, *flags)
        start = time.time()
        for i in tqdm(range(100)):
            m.compute_fit(params)
        end = time.time()
        print(
            'Cached: {}, Riemann: {}, Timed average over 100 runs: {}'.format(
                *flags, (end - start) / 100))


def test_model_computation():

    folder = '../data/Voss-2018-04-29/sounding/'
    sola_file = 'sola_20180331-20180430.nc'
    date = '2018-04-29 11:06:37'

    alt, tmp, pre, dtp = get_observed_data(folder + sola_file, date)

    deltaT = 0.5 * units.kelvin
    T_0 = (273.15 + 8) * units.kelvin
    Td_0 = (273.15 + 0) * units.kelvin

    params = ModelParams(T_0,
                         deltaT,
                         Td_0,
                         dew_point_anomaly=0 * units.kelvin,
                         thermal_altitude=100,
                         drag_coeff=0,
                         entrainment_coeff=0,
                         humidity_entrainment_coeff=0,
                         aspect_ratio=0)

    result = ModelResult()

    m = ThermalModel(alt, pre, tmp, dtp)

    # plt.plot(pre, alt, ',')
    # plt.plot(m.pressure, m.altitude, ',')
    # plt.show()

    start = time.time()
    for i in range(100):
        b, _ = m.calc_buoyancy_profile(params, result)
        w, z, bb = m.calc_velocity_profile(b, alt, 300, debug=True)

    end = time.time()
    print('Timed average over 100 runs:', (end - start) / 100)

    # Riemann
    start = time.time()
    for i in range(100):
        b, _ = m.calc_buoyancy_profile(params, result)
        #    def riemann_calc_velocity_profile(self,  buoyancy, altitude, z_0, alpha=0.00, quadratic_drag=0, w0=0.01):
        w, z = m.riemann_calc_velocity_profile(b, alt, 300)

    end = time.time()
    print('Timed Riemann average over 100 runs:', (end - start) / 100)

    #
    # CACHED TIMES
    m = ThermalModel(alt, pre, tmp, dtp, cached=True)

    start = time.time()
    for i in range(100):
        b, _ = m.calc_buoyancy_profile(params, result)
        w, z, bb = m.calc_velocity_profile(b, alt, 300, debug=True)

    end = time.time()
    print('Cached timed average over 100 runs:', (end - start) / 100)

    # Riemann
    start = time.time()
    for i in range(100):
        b, _ = m.calc_buoyancy_profile(params, result)
        #    def riemann_calc_velocity_profile(self,  buoyancy, altitude, z_0, alpha=0.00, quadratic_drag=0, w0=0.01):
        w, z = m.riemann_calc_velocity_profile(b, alt, 300)

    end = time.time()
    print('Cached riemann timed average over 100 runs:', (end - start) / 100)

    zz = np.linspace(0, 3000, 100000)

    # plt.title('Check spline interpolation')
    # plt.plot(bb.b(zz), zz)
    # plt.plot(b, alt, 'x')
    # plt.xlabel('Buoyancy')
    # plt.xlabel('altitude')
    # plt.show()
    #
    # plt.plot(w, z, 'x-')
    # plt.plot(b * 100, alt)
    # plt.plot(tmp.magnitude - 273.15, alt)
    # plt.show()


def test_model_fitting():

    folder = '../data/Voss-2018-04-29/sounding/'
    sola_file = 'sola_20180331-20180430.nc'
    date = '2018-04-29 11:06:37'

    alt, tmp, pre, dtp = get_observed_data(folder + sola_file, date)

    deltaT = 0.5 * units.kelvin
    p_0 = 1008 * units.hPa
    T_0 = (273.15 + 8) * units.kelvin
    Td_0 = (273.15 + 0) * units.kelvin

    m = ThermalModel(alt, pre, tmp, dtp)
    b, _ = m.calc_buoyancy_profile(T_0, Td_0, deltaT)
    w, z = m.calc_velocity_profile(b, alt, 300)

    df = pd.read_csv('../../data/Voss-vol2/clusters.csv')

    w, z, res = m.fit_thermal(
        df[df.labels == 178],
        [T_0.magnitude, Td_0.magnitude, deltaT.magnitude, 0, 0, 500], [
            (273.15, 293.15),
            (263.15, 283.15),
            (0, 5),
            (0, 0.1),
            (0, 0.1),
            (0, 1000),
        ])

    plt.plot(w, z)

    b, _ = m.calc_buoyancy_profile(res.x[0] * units.kelvin,
                                   res.x[1] * units.kelvin,
                                   res.x[2] * units.kelvin,
                                   gamma=res.x[4])
    w, z = m.calc_velocity_profile(b, alt, res.x[5], alpha=res.x[3])
    plt.plot(w, z)
    plt.show()

    # print (res)
    # print (res.x)


def get_observed_data(
        filename='../../data/Voss-2018-04-29/sounding/sola_20180331-20180430.nc',
        date='2018-04-29 11:06:37'):

    # folder = '../data/Voss-2018-04-29/sounding/'
    # sola_file = 'sola_20180331-20180430.nc'
    # date = '2018-04-29 11:06:37'

    ds = nc.Dataset(filename)
    xds = xr.open_dataset(filename)
    df = xds.to_dataframe()
    data = df.loc[date].iloc[3:, :]

    d = data.altitude[data.altitude < 3000]
    index = data.altitude[data.altitude < 3000].index

    alt = data.altitude[index].values * units[ds.variables['altitude'].units]
    tmp = data.air_temperature[index].values * units[
        ds.variables['air_temperature'].units]
    pre = data.air_pressure[index].values * units[
        ds.variables['air_pressure'].units]
    dtp = data.dew_point_temperature[index].values * units[
        ds.variables['dew_point_temperature'].units]

    return alt, tmp, pre, dtp


###################################################################################################################
#       SAMPLING THE MODEL HERE
###################################################################################################################
def sample_buoyancy_no_temp():

    folder = '../data/Voss-2018-04-29/sounding/'
    sola_file = 'sola_20180331-20180430.nc'
    date = '2018-04-29 11:06:37'

    alt, tmp, pre, dtp = get_observed_data(folder + sola_file, date)

    deltaT = 0.5 * units.kelvin
    T_0 = (273.15 + 8) * units.kelvin
    Td_0 = (273.15 + -4) * units.kelvin

    params = ModelParams(T_0,
                         deltaT,
                         Td_0,
                         dew_point_anomaly=0 * units.kelvin,
                         thermal_altitude=0,
                         drag_coeff=0,
                         entrainment_coeff=0,
                         humidity_entrainment_coeff=0,
                         aspect_ratio=0)
    result = ModelResult()
    m = ThermalModel(alt, pre, tmp, dtp)

    profile_idx = []
    profiles = []

    start = time.time()
    for td in (np.linspace(-5, 5, 10) + 273.15) * units.kelvin:
        for dt in np.linspace(0, 3,
                              10) * units.kelvin:  # TODO: dt 0 is useless
            for g in np.exp(np.linspace(-3, -7, 10)):
                try:
                    params.dew_point_temperature = td
                    params.temperature_anomaly = dt
                    params.entrainment_coeff = g
                    b, _ = m.calc_buoyancy_profile(params,
                                                   result,
                                                   use_sounding=False)
                    profile_idx.append(
                        (T_0.magnitude, td.magnitude, dt.magnitude, g))
                    profiles.append(b.magnitude)
                except RuntimeError as re:
                    print(re)
    end = time.time()
    print('Takes fucking forever:', end - start)

    data = pd.DataFrame(profiles,
                        index=pd.MultiIndex.from_tuples(
                            profile_idx, names=['T', 'dew', 'dT', 'gamma']))
    print(data)

    # data.to_csv('sampled_buoyancy.csv')

    # data.T.plot(legend=None)
    # plt.show()

    # plt.title('Buoyancy')
    # plt.plot(b, alt, 'x')
    # plt.xlabel('Buoyancy')
    # plt.xlabel('altitude')
    # plt.show()


if __name__ == '__main__':

    # sample_buoyancy_no_temp()

    # test_riemann_integration()
    # test_model_computation()
    time_model_computation()
    # test_model_fitting()

    # folder = '../data/Voss-2018-04-29/sounding/'
    # list_dir = os.listdir(folder)
    # sola_file = 'sola_20180331-20180430.nc'
    #
    # # ds = [nc.Dataset(folder + l) for l in list_dir]
    # ds = nc.Dataset(folder + sola_file)
    #
    # xds = xr.open_dataset(folder + sola_file)
    # df = xds.to_dataframe()
    # data = df.loc['2018-04-29 11:06:37'].iloc[3:, :]
    #
    # deltaT = 0.5 * units.kelvin
    # p_0 = 1008 * units.hPa
    # T_0 = (273.15 + 8) * units.kelvin
    # Td_0 = (273.15 + 0) * units.kelvin
    #
    # alt = data.altitude[data.altitude < 3000] * units[ds.variables['altitude'].units]
    # tmp = data.air_temperature[alt.index].values * units[ds.variables['air_temperature'].units]
    # pre = data.air_pressure[alt.index].values * units[ds.variables['air_pressure'].units]
    # dtp = data.dew_point_temperature[alt.index].values * units[ds.variables['dew_point_temperature'].units]
    #
    # res = np.polyfit(alt, np.log(pre.magnitude), 1)
    #
    # plt.plot(alt, np.log(pre.magnitude))
    # plt.plot(alt, res[0] * alt + res[1])
    # plt.show()
    #
    # plt.plot(alt, pre.magnitude)
    # plt.plot(alt, np.exp(res[0] * alt + res[1]))
    # plt.show()