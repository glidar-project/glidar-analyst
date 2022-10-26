import matplotlib
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QVBoxLayout, \
    QComboBox, QApplication, QSizePolicy

from glidar_analyst.model.convection_model import ThermalModel
from glidar_analyst.gui.matplot_vidgets import MplWidget
from glidar_analyst.util.decorators import my_timer

matplotlib.use('Qt5Agg')

from PyQt5 import QtWidgets
import numpy as np

import metpy.calc as mpcalc
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units


class MetpyView(QtWidgets.QWidget):

    day_changed = pyqtSignal(object)

    def set_model(self, model):
        self.model = model

    def set_data(self, nc_data, df_data):
        self.nc_data = nc_data
        self.df_data = df_data
        self.combo_box.clear()
        self.combo_box.addItems([str(d) for d in df_data.index.levels[0]])

    def __init__(self, zmax=5000, *args, **kwargs):
        super(MetpyView, self).__init__(*args, **kwargs)

        self.model = None
        self.model_result = None
        self.nc_data = None
        self.df_data = None

        self.show_humid = False

        self.combo_box = QComboBox()
        self.combo_box.currentTextChanged.connect(self.plot_sounding)

        self.pyplot_view = MplWidget(dpi=100)
        self.pyplot_view.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)
        self.pyplot_view.toolbar.show()
        self.pyplot_view.sc.fig.tight_layout()
        self.pyplot_view.sc.fig.set_size_inches(10.5, 10.5)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.combo_box)
        self.layout.addWidget(self.pyplot_view)

        self.lims = (-10, 25), (1015, 600)
        self.skew = None
        self.model_params = None
        self.params = None
        self.shade = []
        self.flight_data = []
        self.p = None
        self.T = None
        self.z = None
        self.zmax = zmax
        self.entrained = None
        self.cloud_base = 0

        self.add_menu()

        self.empty_skew()

    def add_menu(self):

        mb = self.parent().parent().menuBar()

        m = mb.addMenu('Model')

        toggle = QAction('Toggle Humid', m)
        toggle.triggered.connect(self.toggle_humid)
        m.addAction(toggle)

    def toggle_humid(self):
        self.show_humid = not self.show_humid

        self.plot_model_result(self.model_result)

    def empty_skew(self):

        if self.skew:
            xlim, ylim = self.skew.ax.get_xlim(), self.skew.ax.get_ylim()
        else:
            xlim, ylim = self.lims
        #
        # Skew figure stuff
        fig = self.pyplot_view.sc.fig
        fig.clear()
        add_metpy_logo(fig, -100, -100)
        skew = SkewT(fig, rotation=45)

        # set the axis limits
        skew.ax.set_ylim(*ylim)
        skew.ax.set_xlim(*xlim)
        # skew.ax.set_aspect(200)

        # Add the relevant special lines
        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        # skew.plot_mixing_lines()

        # An example of a slanted line at constant T -- in this case the 0
        # isotherm
        skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

        fd = self.flight_data.copy()
        self.flight_data = []
        for e in fd:
            print(e)
            self.flight_data += skew.ax.plot(e.get_xdata(),
                                             e.get_ydata(),
                                             color=e.get_color(),
                                             linestyle=e.get_linestyle(),
                                             marker=e.get_marker())
        p = np.linspace(1000, 600) * units.hPa
        result = np.ones_like(p) * 270 * units.kelvin

        self.skew = skew

        self.shade = {
            # self.skew.plot(result.lcl_pressure, result.lcl_temperature, 'ko', markerfacecolor='black', label='lcl'),
            # The humidity profiles
            'dew_bar':
            self.skew.plot(p, result, 'b', label='base dew point'),
            'dew_pert':
            self.skew.plot(p,
                           result,
                           'b:',
                           linewidth=2,
                           label='parcel dew point'),
            'dew_ent':
            self.skew.plot(p,
                           result,
                           'b--',
                           linewidth=2,
                           label='entrained parcel dew point'),
            # The dry base profiles
            't_bar':
            self.skew.plot(p,
                           result,
                           '-',
                           color='black',
                           linewidth=2,
                           label='base temperature'),
            't_pert':
            self.skew.plot(p,
                           result,
                           ':',
                           color='black',
                           linewidth=2,
                           label='parcel temperature'),
            't_ent':
            self.skew.plot(p,
                           result,
                           '--',
                           color='black',
                           linewidth=2,
                           label='entrained parcel temperature'),
            # self.skew.plot(self.p, self.prof2, 'k--', linewidth=2, label='dry T pert'),
            # self.skew.plot(self.p, self.prof1, 'k', linewidth=2, label='dry T bar'),
            # The humid temperature profiles
            't_bar_virtual':
            self.skew.plot(p, result, '-', c='grey', label='virtual base temperature'),
            't_pert_virtual':
            self.skew.plot(p, result, ':', c='grey', alpha=1, label='virtual parcel temperature'),
            't_ent_virtual':
            self.skew.plot(p, result, '--', c='grey', alpha=1, label='entrained virtual parcel temperature'),
            'cin':
            self.skew.shade_cin(
                p, result,
                result),  # Convective Inhibition area shaded in blue
            'cape':
            self.skew.shade_cape(
                p, result,
                result),  # Convective Available Potential Energy (CAPE) in red
            'cloud': None
            # [
            #     # self.skew.ax.text(
            #         # self.lcl_temperature + 1 * self.lcl_temperature.units,
            #         # self.lcl_pressure,
            #         # "added water:\n  {:.2f}".format(result.added_water)),
            #         # "added: {:.2f}\ncloud: {:.2f}".format(result.added_water, result.cloud_water)),
            #     # self.skew.ax.fill_betweenx(self.p, result.t_ent, temp, color='g', alpha=0.3),
            #     self.skew.shade_cin(p[zz], base[zz], temp[zz]),           # Convective Inhibition area shaded in blue
            #     self.skew.shade_cape(p[zz], base[zz], temp[zz]),          # Convective Available Potential Energy (CAPE) in red
            #     self.skew.ax.fill_between(
            #         [-100, 100], [result.lcl_pressure, result.lcl_pressure],
            #         [100, 100] * units.hPa, facecolor='lightblue', alpha=0.5)
            #     ],
            # self.skew.plot(p, temp, color='cornflowerblue', label='virtual T pert')
        }

        # Show the plot
        self.pyplot_view.sc.draw()
        self.repaint()

    def resizeEvent(self, event):
        print('resize event happening', event.size())

        self.skew.ax.set_aspect(150 * event.size().height() /
                                event.size().width())

    def plot_flight_temp(self, selection):
        if self.skew is None:
            return

        for e in self.flight_data:
            e.remove()
        self.flight_data = []

        if selection is not None and len(selection) > 0:
            d = selection[0].data
            try:
                self.flight_data += self.skew.plot(
                    d.pressure.values * units.Pa,
                    d.temperature.values * units.celsius, 'r.', markersize=2, label='thermal temperature')
                self.flight_data += self.skew.plot(
                    d.pressure.values * units.Pa,
                    d.dewpoint.values * units.celsius, 'g.', markersize=2, label='thermal dewpoint')
            except Exception as e:
                # print(e)
                pass

        self.pyplot_view.sc.draw()
        self.repaint()

    # @my_timer
    def plot_model_result(self, result):

        if result is None:
            return

        self.model_result = result
    
        zz = result.air_profile.altitude.magnitude > result.params.thermal_altitude

        if self.show_humid:
            temp = result.t_ent_virtual.to('celsius').magnitude[zz]
            base = result.t_bar_virtual.to('celsius').magnitude[zz]
        else:
            temp = result.t_ent.to('celsius').magnitude[zz]
            base = result.t_bar.to('celsius').magnitude[zz]

        # # Check for input values
        # if result.added_water is None:
        #     result.added_water = np.NaN

        # if result.cloud_water is None:
        #     result.cloud_water = np.NaN

        p = result.air_profile.pressure.to('hPa').magnitude

        self.shade['dew_bar'][0].set_data(
            result.dew_bar.to('celsius').magnitude, p)
        self.shade['dew_pert'][0].set_data(
            result.dew_pert.to('celsius').magnitude, p)
        self.shade['dew_ent'][0].set_data(
            result.dew_ent.to('celsius').magnitude, p)

        self.shade['t_bar'][0].set_data(
            result.t_bar.to('celsius').magnitude, p)
        self.shade['t_pert'][0].set_data(
            result.t_pert.to('celsius').magnitude, p)
        self.shade['t_ent'][0].set_data(
            result.t_ent.to('celsius').magnitude, p)

        self.shade['t_bar_virtual'][0].set_data(
            result.t_bar_virtual.to('celsius').magnitude, p)
        self.shade['t_pert_virtual'][0].set_data(
            result.t_pert_virtual.to('celsius').magnitude, p)
        self.shade['t_ent_virtual'][0].set_data(
            result.t_ent_virtual.to('celsius').magnitude, p)

        for k in ['t_bar_virtual', 't_pert_virtual', 't_ent_virtual']:
            self.shade[k][0].set_visible(self.show_humid)

        # Convective Available Potential Energy (CAPE) in red
        if self.shade['cape']:
            self.shade['cape'].remove()
            self.shade['cape'] = None
        if self.shade['cin']:
            self.shade['cin'].remove()
            self.shade['cin'] = None
     
        cin = base >= temp
        self.shade['cin'] = self.skew.ax.fill_betweenx(
                    p[zz][cin], base[cin], temp[cin], facecolor='lightblue', alpha=0.5)

        cape = base <= temp
        self.shade['cape'] = self.skew.ax.fill_betweenx(
                    p[zz][cape], base[cape], temp[cape], facecolor='red', alpha=0.5)
     
        if self.shade['cloud']:
            self.shade['cloud'].remove()
            self.shade['cloud'] = None

        if result.lcl_pressure is not None:
            self.shade['cloud'] = self.skew.ax.fill_between(
                        [-100, 100], [result.lcl_pressure, result.lcl_pressure],
                        [100, 100] * units.hPa, facecolor='lightgrey', alpha=0.5)


        # self.shade = {
        #     # self.skew.plot(result.lcl_pressure, result.lcl_temperature, 'ko', markerfacecolor='black', label='lcl'),
        #     # The humidity profiles
        #     'dew_bar': self.skew.plot(p, result.dew_bar, 'b', label='dew point'),
        #     'dew_pert': self.skew.plot(p, result.dew_pert, 'b:', linewidth=2, label='parcel dew point'),
        #     'dew_ent': self.skew.plot(p, result.dew_ent, 'b--', linewidth=2, label='entrained parcel dew point'),
        #     # The dry base profiles
        #     't_bar': self.skew.plot(p, result.t_bar, '-', color='grey', linewidth=2, label='dry T bar (model)'),
        #     't_ent': self.skew.plot(p, result.t_ent, '--', color='grey', linewidth=2, label='entrained parcel profile'),
        #     # self.skew.plot(self.p, self.prof2, 'k--', linewidth=2, label='dry T pert'),
        #     # self.skew.plot(self.p, self.prof1, 'k', linewidth=2, label='dry T bar'),
        #     # The humid temperature profiles
        #     't_bar_virtual': self.skew.plot(p, result.t_bar_virtual, '-', c='black', label='virtual T bar'),
        #     't_pert_virtual': self.skew.plot(p, result.t_pert_virtual, ':', c='black', alpha=1, label='T pert'),
        #     't_ent_virtual': self.skew.plot(p, result.t_ent_virtual, '--', c='black', alpha=1, label='T ent'),
        #     # [
        #     #     # self.skew.ax.text(
        #     #         # self.lcl_temperature + 1 * self.lcl_temperature.units,
        #     #         # self.lcl_pressure,
        #     #         # "added water:\n  {:.2f}".format(result.added_water)),
        #     #         # "added: {:.2f}\ncloud: {:.2f}".format(result.added_water, result.cloud_water)),
        #     #     # self.skew.ax.fill_betweenx(self.p, result.t_ent, temp, color='g', alpha=0.3),
        #     #     self.skew.shade_cin(p[zz], base[zz], temp[zz]),           # Convective Inhibition area shaded in blue
        #     #     self.skew.shade_cape(p[zz], base[zz], temp[zz]),          # Convective Available Potential Energy (CAPE) in red
        #     #     self.skew.ax.fill_between(
        #     #         [-100, 100], [result.lcl_pressure, result.lcl_pressure],
        #     #         [100, 100] * units.hPa, facecolor='lightblue', alpha=0.5)
        #     #     ],
        #     # self.skew.plot(p, temp, color='cornflowerblue', label='virtual T pert')
        # }

        self.pyplot_view.sc.draw()
        # self.pyplot_view.sc.flush_events()
        # self.repaint()

    def plot_mixing_line(self, temperature, pressure, dew_point_temp):
        """
        Adds a mixing line into the plot,
        requires units with the parameters.

        :param temperature: [kelvin]
        :param pressure: [hPa]
        :param dew_point_temp: [kelvin]
        :return: None
        """
        # todo: this should not be in critical section..
        with ThermalModel.semaphore:
            rel_humidity = mpcalc.relative_humidity_from_dewpoint(
                temperature, dew_point_temp)
        try:
            # mixing_ratio = mpcalc.mixing_ratio_from_relative_humidity(rel_humidity, temperature, pressure)    # This used to fucking work
            mixing_ratio = mpcalc.mixing_ratio_from_relative_humidity(
                pressure, temperature, rel_humidity)  # Now we have to use this
        except ValueError as ve:
            print(ve)
            return

        w = mixing_ratio * np.array([1]).reshape(-1, 1)

        # self.skew.plot_mixing_lines(w=w, p=self.p)            # This used to fucking work
        self.skew.plot_mixing_lines(mixing_ratio=w,
                                    pressure=self.p)  # Now we have to use this

    def plot_day(self, df, p_0=None, T_0=None, Td_0=None, deltaT=None):

        self.empty_skew()

        df = df.dropna(subset=('air_temperature', 'dew_point_temperature',
                               'wind_from_direction', 'wind_speed'),
                       how='all').reset_index(drop=True)

        self.z = df['altitude'].values
        self.p = df['air_pressure'].values * units[
            self.nc_data.variables['air_pressure'].units]
        self.T = df['air_temperature'].values * units[
            self.nc_data.variables['air_temperature'].units]

        Td = df['dew_point_temperature'].values * units[
            self.nc_data.variables['dew_point_temperature'].units]
        wind_speed = df['wind_speed'].values * units[
            self.nc_data.variables['wind_speed'].units]
        wind_dir = df['wind_from_direction'].values * units[
            self.nc_data.variables['wind_from_direction'].units]
        u, v = mpcalc.wind_components(wind_speed, wind_dir)

        # Plot the data using normal plotting functions, in this case using
        # log scaling in Y, as dictated by the typical meteorological plot.
        self.skew.plot(self.p, self.T, 'r', label='observed T')
        self.skew.plot(self.p, Td, 'g', label='observed dewpoint')
        sub_idx = np.arange(u.size)[::20]
        self.skew.plot_barbs(self.p[sub_idx], u[sub_idx], v[sub_idx])

        self.plot_model(self.params)

        self.pyplot_view.sc.draw()
        self.repaint()



    def plot_model(self, params, model=None, T_ent=None):
        """
        Here I need model results to plot, so I'll expect model
        object with the following data:
         - Base RH profile 
         - Perturbed RH profile
         - Base temperature profile
         - Perturbed temperature profile
         - Entrained temperature profile
         - LCL pressure
         - LCL temperature
         - mixing values
        """

        if self.T is None:
            return
        if params is None:
            return

        # params should come with the units set already,
        # such that this nonsense is not necessary
        self.params = params
        T_0 = params['T'] * units.kelvin
        Td_0 = params['Td'] * units.kelvin
        deltaT = params['dT'] * units.kelvin
        deltaTd = params['dTd'] * units.kelvin
        self.z_0 = params['z0']  # lovely fucking duplicity

        pp_voss = self.p  # [self.p<p_0] # TODO: Bug here...
        p_0 = pp_voss[0]
        z = self.z  # [self.p<p_0] # TODO: Bug here...

        # So far this is the best I figured to circumvent
        # parallel access to the metpy library, except this should be only the
        # visualization class so no computation should be necessary
        # todo: get rid of critical section here
        with ThermalModel.semaphore:
            self.lcl_pressure, self.lcl_temperature = mpcalc.lcl(
                p_0, T_0 + deltaT, Td_0)
            # Calculate full parcel profile and add to plot as black line
            self.prof1 = mpcalc.parcel_profile(pp_voss, T_0,
                                               Td_0).to(units.kelvin)
            # Calculate parcel profile for thermal
            self.prof2 = mpcalc.parcel_profile(pp_voss, T_0 + deltaT,
                                               Td_0).to(Td_0.units)

        base = self.z[self.p < self.lcl_pressure]  # can be an empty array
        if base.size > 0:
            self.cloud_base = base[0]
        else:
            self.cloud_base = self.zmax + 1

        # Shade areas of CAPE and CIN
        T = self.T  # [self.p <= p_0] # TODO: Bug here...
        self.prof = np.where(self.prof1 < T, T, self.prof1)

        # for s in self.shade:
        #     for e in s:
        #         e.remove()
        # zz = z > self.z_0

        # self.shade = [
        #     self.skew.plot(self.lcl_pressure, self.lcl_temperature, 'ko', markerfacecolor='black', label='lcl'),
        #     self.skew.plot(pp_voss, self.prof2, 'k--', linewidth=2, label='dry T pert'),
        #     self.skew.plot(pp_voss, self.prof1, 'k', linewidth=2, label='dry T bar'),
        #     [self.skew.shade_cin(pp_voss[zz], self.prof[zz], self.prof2[zz], label='dry CIN'),     # Convective Inhibition area shaded in blue
        #      self.skew.shade_cape(pp_voss[zz], self.prof[zz], self.prof2[zz], label='dry CAPE')]    # Convective Available Potential Energy (CAPE) in red
        # ]

        # self.plot_mixing_line(T_0, p_0, Td_0)

        # Show the plot
        self.pyplot_view.sc.draw()
        self.repaint()

    def plot_sounding(self, name):
        """
        Here we listen to the combobox
        :param name: selected day to plot
        :return: None
        """

        print('Plot sounding: ', name)
        if name is None or not name:
            return

        # Get the corresponding day from the data source
        # Ditch first 3 measurements due to systematical error
        data = self.df_data.loc[name].iloc[3:, :]

        # Plot the observed data only while we wait for model
        # results
        self.plot_day(data[data.altitude < self.zmax])

        # Let everyone know there is new sounding selected
        self.day_changed.emit(data)

        # Here we plot a simplified version of the model
        # Basically just showing the param values
        # No model results has been produced yet...
        params = {
            'dT': 1,
            'T': (10 + 273.15),  # use kelvins
            'Td': (5 + 273.15),  # use kelvins
            'dTd': 0,
            'a': 0,
            'B': 0,
            'z0': 0,
        }
        if self.params:
            self.plot_model(self.params)
        else:
            self.plot_model(params)

    def model_params_changed(self, params):
        self.model_params = params


if __name__ == "__main__":
    import netCDF4 as nc
    import xarray as xr
    import sys
    app = QApplication(sys.argv)

    filename = '../data/Voss-2018-04-29/sounding/' + 'sola_20180331-20180430.nc'

    window = MetpyView(nc.Dataset(filename),
                       xr.open_dataset(filename).to_dataframe())
    window.show()
    sys.exit(app.exec_())
