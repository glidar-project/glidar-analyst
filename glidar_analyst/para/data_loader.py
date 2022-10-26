import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime

from PyQt5.QtWidgets import QFileDialog, QApplication
from tqdm import tqdm

from scipy import optimize
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import maximum_filter, median_filter

from pyproj import Proj
import metpy.calc as mpcalc
from metpy.units import units

from glidar_analyst.para.kml_parser import KmlParser
from glidar_analyst.para.igc_parser import IgcParser
from glidar_analyst.para.skydrop_parser import SkyDropIgcParser

plt.rcParams['figure.figsize'] = (16.0, 9.0)

test_filename = "/home/juraj/Work/ConvectionAnalysis/data/Voss-2018-04-29/2018-04-29_Erik_Hamran_Nilsen.kml"


class TrackAnalysis:

    altitude_sigma = 5
    velocity_sigma = 3

    def __init__(self, track):

        self.utm_zone = self.get_utm_zone(track.track.longitude.iloc[-1])
        self.myProj = Proj(proj='utm', ellps='WGS84',
                           zone=self.utm_zone,
                           units='m')

        # Just gutting the parsed track
        # self.filename = track.filename
        # self.takeoff_time = track.takeoff_datetime
        self.frame = track.track

        # Computation from here
        self.frame['dt'] = np.pad((self.frame.time.values[2:] - self.frame.time.values[:-2]).astype(float), 1, mode='edge') * 1e-9
        z = gaussian_filter1d(self.frame.altitude.values, self.altitude_sigma)
        self.frame['vario'] = np.pad(z[2:] - z[:-2], 1, mode='edge') / self.frame.dt.values

        self.x, self.y = self.myProj(self.frame.longitude.values, self.frame.latitude.values)    # Lon, Lat
        self.frame['x'] = self.x
        self.frame['y'] = self.y

        x = gaussian_filter1d(self.x, self.velocity_sigma)
        y = gaussian_filter1d(self.y, self.velocity_sigma)
        self.frame['dx'] = np.pad(x[2:] - x[:-2], 1, mode='edge') / self.frame.dt.values
        self.frame['dy'] = np.pad(y[2:] - y[:-2], 1, mode='edge') / self.frame.dt.values

        self.thermals = self.segment_thermals_from_track(self.frame)

        # circle_fit_points = int(240 / self.time_step)
        # self.circle_fit = self._circle_fit(self.dx, self.dy, circle_fit_points)
        # self.circle_fit_60 = self._circle_fit(self.dx, self.dy, 60)
        # self.circle_fit_20 = self._circle_fit(self.dx, self.dy, 20)
        #
        # self.curvature = self.curvature_hack(np.stack((self.dx, self.dy)) - self.circle_fit['center'].T)
        # self.curvature_fit = self._curvature_fit(self.x, self.y, 20)
        #
        # self.cc = gaussian_filter1d(self.curvature_fit['radius'], 15)
        # self.dcc = np.pad((self.cc[2:] - self.cc[:-2]), 1, mode='edge') / self.dt


    @staticmethod
    def segment_thermals_from_track(df, median_size=15, max_filter_size=20, minimal_thermal_gain=100):

        positiveVario = df[median_filter(df['vario'], median_size) > 0]

        mask = np.zeros_like(df.index)
        mask[positiveVario.index] = 1

        mask = maximum_filter(mask, max_filter_size)

        d_mask = mask[1:] - mask[:-1]
        d_mask[d_mask == -1] = 0
        labels = np.concatenate([np.array([0]), np.cumsum(d_mask)])

        thermals = pd.DataFrame(positiveVario)
        thermals['labels'] = labels[positiveVario.index]

        gain = thermals.groupby(thermals.labels).altitude.max() - thermals.groupby(thermals.labels).altitude.min()
        labels = gain[gain > minimal_thermal_gain].index

        thermals = [ pd.DataFrame(thermals[thermals.labels == l]) for l in labels ]

        return thermals

    @staticmethod
    def get_utm_zone(lon):

        lon += 180.0
        zone = 1 + int(lon // 6)

        return zone

    # def _curvature_fit(self, x, y, w):
    #
    #     alg_cic = np.array([fit_circle_alg(x[i:i + w], y[i:i + w]) for i in np.arange(x.size - w)])
    #     # alg_cic = [FitCircle((x[i:i + w], y[i:i + w])) for i in np.arange(x.size - w)]
    #     # alg_cic = np.array([(f.R, *f.center) for f in alg_cic])
    #
    #     npad = ((w // 2, w - w // 2), (0, 0))
    #     alg_cic = np.pad(alg_cic, pad_width=npad, mode='edge')
    #
    #     circle_fit_radius = alg_cic[:, 0]
    #     circle_fit_center = alg_cic[:, 1:]
    #
    #     return {'radius': circle_fit_radius, 'center': circle_fit_center}

    # def _circle_fit(self, x, y, w):
    #
    #     #        print(self.filename, x, y, w)
    #     alg_cic = np.array([fit_circle_alg(x[i:i + w], y[i:i + w]) for i in np.arange(x.size - w)])
    #     # alg_cic = [FitCircle((x[i:i + w], y[i:i + w])) for i in np.arange(x.size - w)]
    #     # alg_cic = np.array([(f.R, *f.center) for f in alg_cic])
    #
    #     npad = ((w // 2, w - w // 2), (0, 0))
    #     alg_cic = np.pad(alg_cic, pad_width=npad, mode='edge')
    #
    #     circle_fit_radius = alg_cic[:, 0]
    #     circle_fit_center = alg_cic[:, 1:]
    #
    #     v = gaussian_filter1d(np.stack((self.dx, self.dy)), 3)
    #
    #     # npad = ((1, 1), (0, 0))
    #     # circ_speed = 0.5 * np.pad(circle_fit_center[2:, :] - circle_fit_center[:-2, :], pad_width=npad, mode='edge')
    #
    #     wind_speed = circle_fit_center
    #
    #     estimate_wind_speed = np.sqrt(np.sum(wind_speed ** 2, axis=1))
    #     return {'radius': circle_fit_radius, 'center': circle_fit_center, 'speed': estimate_wind_speed}

    # def curvature_hack(self, D):
    #     """
    #     Computes the curvature of a 2D line
    #     The shape of the array should be (2, n),
    #     where n is the number of points
    #
    #     """
    #
    #     D2 = gaussian_filter1d(D[:, 2:] - D[:, :-2], 3, mode='nearest') / np.stack([self.dt[1:-1], self.dt[1:-1]])
    #
    #     d2 = np.sum(np.power(D, 2), axis=0)
    #     d = np.power(d2, 1.5)
    #
    #     k = np.pad(np.abs(D.T[1:-1, 0] * D2.T[:, 1] - D.T[1:-1, 1] * D2.T[:, 0]) / d[1:-1], 1, mode='edge')
    #     return k

    # def segment_thermals(self, sink_thr=0, radius_threshold=200, radius_smooth=15, min_thermal_gap=10,
    #                      min_thermal_length=30):
    #
    #     idx_vario = np.where((self.vario > sink_thr) &
    #                          (np.abs(self.dcc) < 5) &
    #                          (self.cc < radius_threshold))[0]
    #
    #     dlabels = (idx_vario[1:] - idx_vario[:-1])
    #     labels = np.cumsum(np.where(dlabels < (min_thermal_gap / self.time_step), 0, 1))
    #
    #     stuff = np.unique(labels, return_counts=True, return_index=True)
    #
    #     thermals = dict()
    #     for i in stuff[0][:-1]:
    #         if stuff[2][i] > (min_thermal_length / self.time_step):
    #             thermals[repr(i)] = (idx_vario[stuff[1][i] + 1], idx_vario[stuff[1][i + 1]])
    #     try:
    #         if stuff[2][-1] > (min_thermal_length / self.time_step):
    #             thermals[repr(len(stuff[0]))] = (idx_vario[stuff[1][-1] + 1], idx_vario[-1])
    #     except IndexError:
    #         pass
    #
    #     return thermals


class FitCircle:
    """
    Credit: https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html#Using-scipy.optimize.leastsq
    """

    def __init__(self, points):
        self.x = points[0]
        self.y = points[1]
        center_estimate = np.mean(self.x), np.mean(self.y)
        self.center, self.ier = optimize.leastsq(self.f_2, center_estimate)

        self.Ri = self.calc_R(*self.center)
        self.R = self.Ri.mean()
        self.residuals = np.sum((self.Ri - self.R) ** 2)

    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((self.x - xc) ** 2 + (self.y - yc) ** 2)

    def f_2(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_R(*c)
        return Ri - Ri.mean()


def fit_circle_alg(x, y):
    """
    Credit: https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html#Using-scipy.optimize.leastsq
    """
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv = np.sum(u * v)
    Suu = np.sum(u ** 2)
    Svv = np.sum(v ** 2)
    Suuv = np.sum(u ** 2 * v)
    Suvv = np.sum(u * v ** 2)
    Suuu = np.sum(u ** 3)
    Svvv = np.sum(v ** 3)

    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0

    try:
        uc, vc = np.linalg.solve(A, B)
    except np.linalg.LinAlgError as e:
        # print (e)
        # print (x)
        # print (y)
        return -1, 0, 0

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calcul des distances au centre (xc_1, yc_1)
    Ri_1 = np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
    R_1 = np.mean(Ri_1)
    # residu_1 = np.sum((Ri_1 - R_1) ** 2)

    return R_1, xc_1, yc_1


def load_data(folder=None):

    if folder is None:
        folder = '../data/Voss-vol2'

    print(os.getcwd(), folder)
    file_list = os.listdir(folder)

    raw_data_kml = [KmlParser(os.path.join(folder, file)) for file in tqdm(file_list[:]) if file[-4:].lower() == '.kml']
    raw_data_igc = [IgcParser(os.path.join(folder, file)) for file in tqdm(file_list[:]) if file[-4:].lower() == '.igc']

    raw_data = [TrackAnalysis(t) for t in tqdm(raw_data_igc + raw_data_kml)]

    if len(raw_data) < 1:
        print('No tracks present.')
        return

    frames = [ data.frame for data in raw_data ]

    dataFrame = pd.concat(frames, ignore_index=True, sort=False)

    return dataFrame


def segment_thermals(dataFrame):

    from sklearn.cluster import DBSCAN
    positiveVario = dataFrame[dataFrame['vario'] >= 0]
    fit = pd.DataFrame(positiveVario[['x', 'y', 'time_sec']])
    fit['time_sec'] = fit['time_sec'] * 0.5
    clustering = DBSCAN(eps=15, min_samples=3).fit(fit)
    positiveVario = pd.DataFrame(positiveVario)
    positiveVario['labels'] = clustering.labels_

    return positiveVario


def compute_metpy_analysis(dataFrame):

    dataFrame['theta'] = mpcalc.potential_temperature(dataFrame.pressure.values * units.pascal,
                                                      dataFrame.temperature.values * units.celsius)

    dataFrame['dewpoint'] = mpcalc.dewpoint_from_relative_humidity(
        dataFrame.temperature.values * units.celsius, dataFrame.RH.values * units.percent
    )

    dataFrame['virtual'] = mpcalc.virtual_potential_temperature(
        dataFrame.pressure.values * units.pascal,
        dataFrame.temperature.values * units.celsius,
        mpcalc.mixing_ratio_from_relative_humidity(
            dataFrame.pressure.values * units.pascal,
            dataFrame.temperature.values * units.celsius,
            dataFrame.RH.values
        )
    )

    return dataFrame


def parse_and_segment(file):

    dataset = None

    if file[-4:].lower() == '.igc':
        parser = SkyDropIgcParser()
        dataset = parser.parse_file(file)
        dataset.to_csv(file[:-4] + '_parsed.csv')

    elif file[-4:].lower() == '.kml':
        parser = KmlParser(file)

    if dataset is not None:

        parser.track = compute_metpy_analysis(parser.track)

        thermals = pd.concat(TrackAnalysis(parser).thermals)
        thermals.to_csv(file[:-4] + '_thermals.csv')


def load_and_segment(folder):
    return segment_thermals(load_data(folder))


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)

    dialog = QFileDialog()
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilter('IGC (*.igc *.IGC)')

    # dialog = QFileDialog()
    # dialog.setAcceptMode(QFileDialog.AcceptOpen)
    # # dialog.setFileMode(QFileDialog.Directory)
    # dialog.setNameFilter('CSV (*.csv *.CSV)')
    # # dialog.setDirectory('..')

    if dialog.exec():
        fname = dialog.selectedFiles()[0]
        print("Parsing folder: ", fname)
        parse_and_segment(fname)
        # dataset.to_csv(fname + '_thermals.csv')
