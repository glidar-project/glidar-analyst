import time
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QAction, QGridLayout, QSizePolicy
from numpy.linalg import LinAlgError

from glidar_analyst.iso_trotter import IsoTrotter, TracerObjectDatabase
from glidar_analyst.model.convection_model import ThermalModel, ModelParams
from glidar_analyst.gui.matplot_vidgets import MplWidget
from glidar_analyst.gui.metpy_view import MetpyView
from glidar_analyst.gui.iso_tracer import ClickFitter, IsoTracer, TracerParamObject
from glidar_analyst.gui.model_controls_widget import ModelControlsWidget
from glidar_analyst.gui.thermal_strip import ThermalsStrip
from glidar_analyst.worker import Worker
from glidar_analyst.gui.my_base_widget import MyBaseWidget
from glidar_analyst.util.decorators import my_timer

from glidar_analyst.model.physics import fit_pressure_altitude
from PyQt5 import QtWidgets, QtGui

import numpy as np
from metpy.units import units


class ModelProxy:

    def __init__(self, parent):

        self.parent = parent

        #   DON'T FUCK WITH UNITS HERE
        #   If you want to change the default units, go to 
        #   model_controls_widget and change it there.
        #
        from metpy.units import units
        sliders = {
            'dT': ('dT', (0, 5), 0.5, 'Temperature anomaly', units.delta_degC),
            'T':
            ('T', (273.15, 293.15), 283.15, 'Air temperature', units.kelvin),
            'dTd': ('dTd', (-5, 5), 0, 'Dew point anomaly', units.delta_degC),
            'Td': ('Td', (263.15, 283.15), 273.15, 'Dew point temperature',
                   units.kelvin),
            'B': ('B', (0, 0.003), 0, 'Entrainment coefficient',
                  (1 / units.m).units),
            'C': ('C', (0, 0.001), 0, 'Humidity Entrainment',
                  (1 / units.m).units),
            'aspect':
            ('aspect', (0, 5), 0, 'Aspect ratio', units.dimensionless),
            'a': ('a', (0, 0.01), 0, 'Linear Drag coefficient',
                  (1 / units.second).units),
            'q': ('q', (0, 0.01), 0, 'Quadratic Drag coefficient',
                  (1 / units.m).units),
            'z0': ('z0', (0, 3000), 0, 'Thermal min altitude', units.meter),
        }

        self.short_names = {
            'dT': 'Delta T',
            'T': 'Temp',
            'dTd': 'Delta Dew',
            'Td': 'Dewpoint',
            'B': 'Entr.',
            'C': 'Humid e.',
            'aspect': 'Aspect',
            'a': 'Drag',
            'q': 'Drag^2',
            'z0': 'z_0',
        }

        self.params = list(sliders.keys())
        self.ranges = {k: v[1] for k, v in sliders.items()}
        self.defaults = {k: v[2] for k, v in sliders.items()}
        self.names = {k: v[3] for k, v in sliders.items()}
        self.units = {k: v[4] for k, v in sliders.items()}

    @staticmethod
    def translate_params(args):
        params = ModelParams(
            surface_temperature=args['T'] * units.kelvin,
            temperature_anomaly=args['dT'] * units.kelvin,
            dew_point_temperature=args['Td'] * units.kelvin,
            dew_point_anomaly=args['dTd'] * units.kelvin,
            thermal_altitude=args['z0'],  # TODO: units for altitude
            drag_coeff=args['a'],
            entrainment_coeff=args['B'],
            humidity_entrainment_coeff=args['C'],
            aspect_ratio=args['aspect'],
            quadratic_drag_coeff=args['q'])
        return params

    @staticmethod
    def translate_params_back(params):
        args = dict()
        args['T'] = params.surface_temperature.magnitude
        args['dT'] = params.temperature_anomaly.magnitude
        args['Td'] = params.dew_point_temperature.magnitude
        args['dTd'] = params.dew_point_anomaly.magnitude
        args['z0'] = params.thermal_altitude
        args['a'] = params.drag_coeff
        args['B'] = params.entrainment_coeff
        args['C'] = params.humidity_entrainment_coeff
        args['aspect'] = params.aspect_ratio
        args['q'] = params.quadratic_drag_coeff
        return args

    def solve(self, params):

        w, z, (t_ent,
               altitude), res = self.parent.compute_derivative_fit(params)

        return z, w


class ModelFittingWidget(MyBaseWidget):

    def __init__(self, *args, **kwargs):

        super(ModelFittingWidget, self).__init__(*args, **kwargs)

        self.model_proxy = ModelProxy(self)
        self.tracer_db = TracerObjectDatabase()
        self.iso_tortter = IsoTrotter(self.model_proxy)

        ##################################################
        # Model Controller Setup
        self.model_controller = ModelControlsWidget(self.model_proxy)
        self.model_controller.paramsChanged.connect(self.model_changed)

        for s in self.model_controller.sliders.values():
            s.enterSignal.connect(self.show_derivative_of_param)
            s.leaveSignal.connect(self.hide_derivative_of_param)

        ##################################################
        # Model View Setup
        self.model = None
        self.nc_data = None
        self.main_model_view = MplWidget(dpi=100)
        self.main_model_view.toolbar.show()
        self.main_model_view.setSizePolicy(QSizePolicy.Expanding,
                                           QSizePolicy.Expanding)

        # self.main_model_view.selectionChanged.connect(
        #     self.show_point_selection)

        self.model_plot_object = []
        self.old_model_plot_object = []
        self.temp_model_plot_object = []
        self.shadow_plot_object = []

        ax = self.main_model_view.sc.axes
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2500)
        self.main_model_view.sc.axes.set_xlabel('vertical velocity [m/s]')
        self.main_model_view.sc.axes.set_ylabel('altitude [m]')

        ##################################################
        # MetPy View Setup
        self.metpy_view = MetpyView(parent=self)
        self.metpy_view.setSizePolicy(QSizePolicy.Expanding,
                                      QSizePolicy.Expanding)
        self.metpy_view.day_changed.connect(self.make_model)

        ##################################################
        # Click Fitting Menu Setup

        self.click_fitter = ClickFitter(self, self.model_proxy,
                                        self.main_model_view)

        ##################################################
        # Iso Tracer
        self.iso_tracer = IsoTracer(model=self.model_proxy,
                                    model_controls=self.model_controller)
        self.iso_tracer.show()

        self.iso_tracer.param_points_hovered.connect(
            self.show_hovered_over_result)
        self.iso_tracer.tracer_point_selected.connect(
            self.change_to_selected_result)
        self.iso_tracer.tracer_points_selected.connect(
            self.show_selected_results)

        ##################################################
        # Thermal Strip Setup
        self.thermal_strip = ThermalsStrip()
        self.thermal_strip.thermal_selection_changed.connect(
            self.plot_thermal_data)

        ##################################################
        # Layout management
        self.setLayout(QGridLayout(self))
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.main_model_view)
        splitter.addWidget(self.metpy_view)
        self.layout().addWidget(splitter, 0, 0, 2, 3)
        self.layout().addWidget(self.thermal_strip, 2, 0, 1, 3)

        ##################################################
        # Class attributes
        self.derivatives = None
        self.derivative_profiles = None
        self.sex = None
        self.last_worker = None

        self.thermal3d = None
        self.thermal_df = None

        self.w = None
        self.z = None

        self.model_result_line = None
        self.cloud_plot_object = None
        self.ground_plot_object = None
        self.text_plot_object = []
        self.ci_plot_object = []
        self.hover_plot_object = []
        self.selected_plot_objects = []

        self.history = []

        self.zmax = 5000

        ##################################################
        # File menu
        # fm = self.parent().menuBar().addMenu('Edit')

        # undo = QAction('Undo', fm)
        # undo.triggered.connect(self.undo_action)
        # fm.addAction(undo)

    def get_params(self):
        """
        Interface for the ClickFitter
        :return: Currently set params
        """
        return self.model_controller.get_params()

    # def set_params(self, params):
    #     """
    #     Interface for the ClickFitter
    #     :param params: fitted params
    #     :return: None
    #     """
    #     self.model_controller.set_params(params)
    #     self.plot_model(self.compute_fit(params), final=True)
    #     self.recompute_derivatives(params)

    def click_fitting_targets(self, targets, variable):
        """
        Interface for the ClickFitter
        :param targets: list of constraints of the model
        :return: None
        """
        self.solve_for_new_targets(targets, variable)
        self.update_control_lock()

    def update_control_lock(self):
        for e in self.model_controller.sliders.values():
            e.setDisabled(False)

        for t in self.click_fitter.targets.keys():
            self.model_controller.sliders[t].setDisabled(True)

    def solve_for_new_targets(self, targets, variable):

        try:
            params = self.iso_tortter.isotrotting(self.get_params(), targets)
            self.click_fitter.confirm_anchor(variable)
            self.new_result(params, variable)
            self.recompute_derivatives(params)

        except (RuntimeError, LinAlgError) as le:

            self.click_fitter.cancell_anchor(variable)
            print(le)

    def set_sounding_data(self, data_nc, data_df):
        self.nc_data = data_nc
        self.metpy_view.set_data(data_nc, data_df)

    def set_thermal_data(self, thermal_df):
        self.thermal_df = thermal_df
        self.thermal_strip.create_tiles(thermal_df)

    def show_point_selection(self, idx):
        if self.thermal3d:
            self.thermal3d.select_points(idx)

    def undo_action(self):

        print('Undo action.')
        if len(self.history) > 1:
            print('Reverting last run...')

            self.history.pop()
            res = self.history.pop()
            self.plot_model(res, final=True)
            self.model_controller.set_params(
                ModelProxy.translate_params_back(res.params))
            print('Done.')

    # @property
    # def click_fit_param(self):
    #     return self._click_fit_param
    #
    # @click_fit_param.setter
    # def click_fit_param(self, val):
    #     if self._click_fit_param:
    #         self.model_controller.sliders[self._click_fit_param].setDisabled(False)
    #     self._click_fit_param = val
    #     if self._click_fit_param:
    #         self.model_controller.sliders[self._click_fit_param].setDisabled(True)

    # def show_context_menu(self, pos):
    #     self.click_fit_menu.exec(pos)

    def plot_hPa_axis(self, data):

        alt = data.altitude[data.altitude < self.zmax]
        p = data.air_pressure[alt.index].values
        a = alt.values

        if self.sex:
            self.sex.remove()

        self.converter = fit_pressure_altitude(p, a)

        self.sex = self.main_model_view.sc.axes.secondary_yaxis('right',
                                                                functions=(self.converter.p,
                                                                           self.converter.a))
        self.sex.set_ylabel('pressure [hPa]')
        self.main_model_view.sc.draw()
        # self.main_model_view.sc.fig.tight_layout()

    def new_result(self, params, variable):

        res = self.compute_fit(params)

        tracer = self.tracer_db.create(
            TracerParamObject(params=params,
                              result=res,
                              targets=self.click_fitter.get_targets(),
                              variable=variable))

        self.iso_tracer.paint_param_point(tracer)
        self.model_controller.set_params(tracer.params)
        self.plot_model(res)

    def model_changed(self, final, params, variable):

        if not self.model:
            return

        try:
            p = self.iso_tortter.isotrotting(params,
                                             self.click_fitter.get_targets())
            self.new_result(p, variable)
            if final:
                self.recompute_derivatives(p)

        except (RuntimeError, LinAlgError) as e:
            print(e)
            # self.iso_tracer.reset_to_old_result()
            ids = list(self.tracer_db.data.keys())
            if len(ids) > 0:
                self.change_to_selected_result(ids[-1])
                self.click_fitter.flash_targets(
                )  # function call order important

    def plot_thermal_data(self, selection):

        self.metpy_view.plot_flight_temp(selection)

        if self.main_model_view.ploted_stuff:
            for t in self.main_model_view.ploted_stuff:
                t.remove()
            self.main_model_view.ploted_stuff = []

            for e in self.text_plot_object:
                e.remove()
            self.text_plot_object = []

            for e in self.ci_plot_object:
                e.remove()
            self.ci_plot_object = []
            self.main_model_view.sc.draw()

        if not selection:
            if self.thermal3d:
                pass
                # self.thermal3d.showThermal(self.thermal_df)
            return

        if self.thermal3d:
            self.thermal3d.showThermal(selection[0].data)

        if self.ground_plot_object:
            self.ground_plot_object.remove()
            self.ground_plot_object = None

        for t in selection:
            print(t)
            cmap = plt.get_cmap('tab20c')

            self.ci_plot_object += self.main_model_view.sc.axes.plot(
                t.data['vario'] + 1.37 + 0.5,
                t.data['altitude'],
                'o',
                color=cmap(0))
            self.main_model_view.plot(t.data['vario'] + 1.37,
                                      t.data['altitude'],
                                      'o',
                                      color=cmap(20. / 255))
            self.ci_plot_object += self.main_model_view.sc.axes.plot(
                t.data['vario'] + 1.37 - 0.5,
                t.data['altitude'],
                'o',
                color=cmap(40 / 255))

            a = t.data['altitude'].to_numpy()
            v = t.data['vario'].to_numpy()
            perm = np.argsort(a)
            # self.main_model_view.sc.axes.fill_betweenx(a[perm], v[perm]  + 1.37, v[perm] + 1.37 - 0.5)

            self.text_plot_object = [
                self.main_model_view.sc.axes.text(
                    8, 2400, str(t.data.time.min().time())),
                self.main_model_view.sc.axes.text(
                    8, 2300, str(t.data.time.max().time()))
            ]

            if self.thermal3d:
                idx = t.data.altitude.idxmin()
                alt = self.thermal3d.get_altitude(t.data.x[idx], t.data.y[idx])
                self.ground_plot_object = self.main_model_view.sc.axes.fill_between(
                    [0, 10], [alt, alt], facecolor='brown', alpha=0.3)

        self.main_model_view.sc.draw()

    def solve_derivative_constraints(self, targets, result, derivatives):

        if len(targets) < 1:
            return derivative

        ranges = self.model.ranges

        der = derivatives[0]

        n = len(targets)
        M = np.zeros((n, n))
        Y = np.zeros((n, 1))
        Y_frac = np.zeros((n, 1))

        for i, vi in enumerate(targets.values()):

            y = np.interp(vi.target_x, *der)
            Y[i, 0] = y - vi.target_y
            # Y_frac[i,0] = Y[i,0] / vi.target_y

        if np.max(np.abs(Y_frac)) < self.epsilon:
            return der

        for i, vi in enumerate(targets.values()):
            for j, vj in enumerate(targets.values()):
                M[i, j] = np.interp(vi.target_x, *
                                    derivatives[vj.variable]) - vi.target_y

        try:
            X = np.linalg.solve(M, Y)
        except LinAlgError as lae:
            print(lae)
            return der

        for i, v in enumerate(targets.values()):
            p[v.variable] += X[i, 0]
            if p[v.variable] > ranges[v.variable][1]:
                p[v.variable] = ranges[v.variable][1]
            if p[v.variable] < ranges[v.variable][0]:
                p[v.variable] = ranges[v.variable][0]

    def change_to_selected_result(self, tracing_obj_id):

        tracing_obj = self.tracer_db.find_by_id(tracing_obj_id)

        self.model_controller.set_params(tracing_obj.params)
        self.plot_model(tracing_obj.result, final=True)
        self.click_fitter.paint_result(tracing_obj)
        # self.click_fitting_targets(tracing_obj.targets)
        self.update_control_lock()

    # @my_timer
    def show_hovered_over_result(self, tracing_objs):

        for e in self.hover_plot_object:
            e.remove()
        self.hover_plot_object = []

        for tid in tracing_objs:
            t = self.tracer_db.find_by_id(tid)
            self.hover_plot_object += self.main_model_view.sc.axes.plot(
                *t.result.virtual_velocity_profile, 'r-', linewidth=1.5)
        self.main_model_view.sc.draw()

    def show_selected_results(self, obj_ids, mapper):

        for e in self.selected_plot_objects:
            e.remove()
        self.selected_plot_objects = []

        for tid in obj_ids:
            t = self.tracer_db.find_by_id(tid)
            self.selected_plot_objects += self.main_model_view.sc.axes.plot(
                *t.result.virtual_velocity_profile,
                c=mapper.get_mpl_color(tid),
                linewidth=1.5)
        self.main_model_view.sc.draw()

    def show_derivative_of_param(self, paramName):
        # print('showing', paramName)
        if self.derivatives is None:
            return
        for e in self.derivatives[paramName]:
            e.set_visible(True)
        self.main_model_view.sc.draw()

    def hide_derivative_of_param(self, paramName):
        if self.derivatives is None:
            return
        for e in self.derivatives[paramName]:
            e.set_visible(False)
        self.main_model_view.sc.draw()

    def recompute_derivatives(self, params):

        def working_method(params, ranges):

            # print(params, ranges, self, self.compute_fit)

            coolwarm = plt.get_cmap('coolwarm')
            colors = coolwarm(np.linspace(0, 255, 2).astype(int))

            result = {}
            for k, p in params.items():
                print('computing derivative for', k)
                r = ranges[k]
                step = 0.01 * (r[1] - r[0])
                paracopy = dict(params)
                result[k] = []
                for i, c in zip([-5, 5], colors):
                    if worker.cancelled:
                        raise RuntimeWarning('Aborting computation.')
                    paracopy[k] = i * step + p
                    if r[0] < paracopy[k] < r[1]:
                        w, z, b, _ = self.compute_derivative_fit(paracopy)
                        result[k].append((w, z, c))

            return result

        def result_method(result):
            derivatives = {}

            # if worker is not self.last_worker:
            #     return

            # self.last_worker = None

            for k, v in result.items():
                derivatives[k] = []
                for w, z, c in v:
                    derivatives[k].extend(
                        self.main_model_view.sc.axes.plot(w, z, color=c))
                    if z[-1] > self.z[-1]:
                        ww = np.interp(z, self.z, self.w)
                        derivatives[k].append(
                            self.main_model_view.sc.axes.fill_betweenx(
                                z, w, ww, alpha=0.7, color=c))
                    else:
                        ww = np.interp(self.z, z, w)
                        derivatives[k].append(
                            self.main_model_view.sc.axes.fill_betweenx(
                                self.z, self.w, ww, alpha=0.7, color=c))

            self.derivatives = derivatives
            for k in derivatives:
                self.hide_derivative_of_param(k)

        def cleanup_method():
            if worker is self.last_worker:
                self.last_worker = None

        # Check if we already have at least one valid result
        if self.model is None or self.z is None:
            return

        if self.derivatives is not None:
            for v in self.derivatives.values():
                for e in v:
                    e.remove()

        self.derivatives = None

        ranges = self.model_controller.get_ranges()
        worker = Worker(working_method, params, ranges)
        # self.last_worker = worker
        worker.signals.result.connect(result_method)
        worker.signals.finished.connect(cleanup_method)
        # MainWindow.threadpool.start(worker)
        result_method(working_method(params, ranges))

    def fit_and_update(self, ww, zz, var):
        if not var:
            return
        p = self.fit_temperature_to(ww, zz, var)
        self.model_controller.sliders[var].set_value(p[var])
        self.plot_model(self.compute_fit(p))
        return p

    def make_model(self, data):

        if self.nc_data is None:
            return

        ds = self.nc_data
        index = data.altitude[data.altitude < self.zmax].index
        alt = data.altitude[index].values * units[
            ds.variables['altitude'].units]
        tmp = data.air_temperature[index].values * units[
            ds.variables['air_temperature'].units]
        pre = data.air_pressure[index].values * units[
            ds.variables['air_pressure'].units]
        dew = data.dew_point_temperature[index].values * units[
            ds.variables['dew_point_temperature'].units]

        self.plot_hPa_axis(data)
        self.model = ThermalModel(alt, pre, tmp, dew)

        self.metpy_view.set_model(self.model)

    def compute_derivative_fit(self, params):
        if self.model is None:
            return
        p = ModelProxy.translate_params(params)
        res = self.model.compute_only_humid_fit(p)
        if res.virtual_velocity_profile:
            w, z = res.virtual_velocity_profile
        return w, z, (res.t_ent, self.model.altitude), res

    # @my_timer
    def compute_fit(self, args):

        if self.model is None:
            print('computing fit, no model present! aborting.')
            return None

        params = ModelProxy.translate_params(args)
        res = self.model.compute_fit(params)

        return res

    def plot_cloud_base(self, result):

        if self.cloud_plot_object:
            self.cloud_plot_object.remove()
            self.cloud_plot_object = None

        if self.converter is None:
            return

        if len(result.virtual_velocity_profile[1]) < 1:
            return
        
        if result.lcl_pressure is None:
            return

        base = self.converter.a(result.lcl_pressure.magnitude)
        top = result.virtual_velocity_profile[1][-1]

        if base > top:
            return

        self.cloud_plot_object = self.main_model_view.sc.axes.fill_between(
            [0, 10], 
            [base, base],
            [top, top],
            facecolor='lightgray',
            alpha=0.5)


    def plot_shadow(self, result):

        if self.shadow_plot_object is not None:
            for e in self.shadow_plot_object:
                e.remove()
        self.shadow_plot_object = []

        if result is None:
            return

        if result.virtual_velocity_profile:
            self.shadow_plot_object = self.main_model_view.sc.axes.plot(
                *result.virtual_velocity_profile,
                color='tab:red',
                linewidth=3,
                alpha=0.3,
                label='humid model shadow')

    # @my_timer
    def plot_vertical_velocity(self, result):

        self.plot_cloud_base(result)

        if result.virtual_velocity_profile:
            self.w = result.virtual_velocity_profile[0]
            self.z = result.virtual_velocity_profile[1]

            if self.model_result_line is None:
                self.model_result_line = self.main_model_view.sc.axes.plot(
                    *result.virtual_velocity_profile,
                    color='orange',
                    linewidth=3,
                    alpha=1,
                    label='humid model')[0]

            else:
                self.model_result_line.set_data(
                    *result.virtual_velocity_profile)

    # if final:
    # if len(self.history) > 0:
    #     self.plot_shadow(self.history[-1])
    # self.history.append(result)

    # if self.model_plot_object is not None:
    #     for e in self.model_plot_object:
    #         e.remove()
    # self.model_plot_object = []

    # if result.virtual_velocity_profile:
    #     self.w = result.virtual_velocity_profile[0]
    #     self.z = result.virtual_velocity_profile[1]

    #     self.model_plot_object = self.main_model_view.sc.axes.plot(
    #         *result.virtual_velocity_profile,
    #         color='orange',
    #         linewidth=3,
    #         alpha=1,
    #         label='humid model')

    # if result.velocity_profile:
    #     self.model_plot_object += self.main_model_view.sc.axes.plot(
    #         *result.velocity_profile, '--',
    #         color='yellow',
    #         linewidth=2,
    #         label='dry model')

        self.main_model_view.sc.draw()

    # @my_timer
    def plot_model(self, result, final=False):

        self.plot_vertical_velocity(result)
        self.metpy_view.plot_model_result(result)
