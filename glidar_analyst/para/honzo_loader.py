import pandas as pd
import xml.etree.ElementTree as ET
import datetime
import numpy as np
import os
from pyproj import Proj

import metpy.calc as mpcalc
from metpy.units import units
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
from tqdm.cli import tqdm
import ctypes


def get_utm_zone(lon):

    lon += 180.0
    zone = 1 + int(lon // 6)

    return zone


def parse_honzo(file):
    tree = ET.parse(file)

    data_labels_ver1 = [
        'temperature'
        , 'humidity'
        , 'pressure'
        , 'altitude'
        , 'vario_'
        , 'G'
        , 'speed_validity'
        , 'heading_validity'
        , 'speed'
        , 'heading'
        , 'caluculated_wind'
        , 'wind_direction'
        , 'circle_fit_x'
        , 'circle_fit_y'
        , 'circle_radius'
        , 'acceleration_x'
        , 'acceleration_y'
        , 'acceleration_z'
        , 'gyro_x'
        , 'gyro_y'
        , 'gyro_z'
        # % ----- The new stuff ----
        , 'mx'
        , 'my'
        , 'mz'
        , 'mx_cor'
        , 'my_cor'
        , 'mz_cor'
        # ,% ----- The new stuff ends. ----
        , 'acceleration_avg_x'
        , 'acceleration_avg_y'
        , 'acceleration_avg_z'
        , 'gyro_avg_x'
        , 'gyro_avg_y'
        , 'gyro_avg_z'
        , 'yaw'
        , 'pitch'
        , 'roll'
        , 'gorund_level'
        # ,% ----- The new stuff ----
        , 'ground_level_interpol'
        # ,% ----- The new stuff ends. ----
        , 'valid_date'
        , 'valid_time'
    ]

    data_labels_ver1_1 = [
        'temperature'
        , 'humidity'
        , 'pressure'
        , 'altitude'
        , 'vario_'
        , 'G'
        , 'speed_validity'
        , 'heading_validity'
        , 'speed'
        , 'heading'
        , 'caluculated_wind'
        , 'wind_direction'
        , 'circle_fit_x'
        , 'circle_fit_y'
        , 'circle_radius'
        , 'acceleration_x'
        , 'acceleration_y'
        , 'acceleration_z'
        , 'gyro_x'
        , 'gyro_y'
        , 'gyro_z'
        # % ----- The new stuff ----
        , 'mx'
        , 'my'
        , 'mz'
        , 'mx_cor'
        , 'my_cor'
        , 'mz_cor'
        # ,% ----- The new stuff ends. ----
        , 'acceleration_avg_x'
        , 'acceleration_avg_y'
        , 'acceleration_avg_z'
        , 'gyro_avg_x'
        , 'gyro_avg_y'
        , 'gyro_avg_z'
        , 'yaw'
        , 'pitch'
        , 'roll'
        , 'gorund_level'
        , 'valid_date'
        , 'valid_time'
    ]

    data_labels_ver2 = [
        'temperature'
        , 'humidity'
        , 'pressure'
        , 'altitude'
        , 'vario_'
        , 'G'
        , 'speed_validity'
        , 'heading_validity'
        , 'speed'
        , 'heading'
        , 'caluculated_wind'
        , 'wind_direction'
        , 'circle_fit_x'
        , 'circle_fit_y'
        , 'circle_radius'
        , 'acceleration_x'
        , 'acceleration_y'
        , 'acceleration_z'
        , 'gyro_x'
        , 'gyro_y'
        , 'gyro_z'
        , 'acceleration_avg_x'
        , 'acceleration_avg_y'
        , 'acceleration_avg_z'
        , 'gyro_avg_x'
        , 'gyro_avg_y'
        , 'gyro_avg_z'
        , 'yaw'
        , 'pitch'
        , 'roll'
        , 'gorund_level'
        , 'valid_date'
        , 'valid_time'
    ]

    data_labels_ver3 = [
        "temperature",
        "humidity",
        "pressure",
        "altitude",
        'vario_',
        'G',
        'speed_validity',
        'heading_validity',
        'speed',
        'heading'
        , 'caluculated_wind'
        , 'wind_direction'
        , 'circle_fit_x'
        , 'circle_fit_y'
        , 'circle_radius'
        , 'yaw'
        , 'pitch'
        , 'roll'
        , 'gorund_level'
        , 'ground_level_interpol'
        , 'valid_date'
        , 'valid_time'
    ]

    root = tree.getroot()

    agg = []

    flight_id = ctypes.c_size_t(file.__hash__()).value

    for child in root[0][0]:
        lat = float(child.attrib['lat'])
        lon = float(child.attrib['lon'])

        # alt = float(child.find('{*}ele').text)
        # note = child.find('{*}note').text
        # time_str = child.find('{*}time').text[:-1]
        # alt = float(child.find('ele').text)
        # note = child.find('note').text
        # time_str = child.find('time').text[:-1]
        assert 'ele' in child[0].tag
        alt = float(child[0].text)
        assert 'note' in child[1].tag
        note = child[1].text
        assert 'time' in child[2].tag
        time_str = child[2].text[:-1]
        try:
            time = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            try:
                time = datetime.datetime.strptime(time_str, "%y-%m-%dT%H:%M:%S")
            except ValueError:
                print('Could not parse a time string:', time_str)
                # time = datetime.datetime(2000, 1, 1)
                continue

        data = [float(v) for v in note.split(',')[1:-1]]

        agg.append((flight_id, time, lat, lon, alt, *data))

    header = ['flight_id', 'time', 'lat', 'lon', 'alt']

    if len(agg) < 2:
        raise RuntimeError('No tracklog produced', file)

    if len(agg[0]) == 27:
        header = header + data_labels_ver3
    elif len(agg[0]) == 38:
        header = header + data_labels_ver2
    elif len(agg[0]) == 45:
        header = header + data_labels_ver1
    elif len(agg[0]) == 44:
        header = header + data_labels_ver1_1
    else:
        raise RuntimeError('Unknown dataformat: {}'.format(len(agg[0]) - 7), file, note)

    frame = pd.DataFrame(agg, columns=header)

    frame['pressure'] = 0.0001 * frame['pressure']  # hPa
    frame['temperature'] = 0.01 * frame['temperature']  # Celsius
    # frame['humidity'] = 0.01 * frame['humidity']    # percent
    while frame.humidity.max() > 100:
        frame.humidity = frame.humidity * 0.1
    frame['dewpoint'] = mpcalc.dewpoint_from_relative_humidity(frame.temperature.values * units.celsius,
                                                               frame.humidity.values * units.percent)
    frame['vario_'] = 0.01 * df['vario_']       # m/s

    # frame['seconds'] = (frame['time'] - frame['time'][0]).apply(lambda d: d.seconds)
    # frame['dt'] = np.pad((frame.seconds.to_numpy()[2:] - frame.seconds.to_numpy()[:-2]), 1, 'edge')
    # z = gaussian_filter1d(frame.alt.values, 10)
    # frame['vario_est'] = np.pad((z[2:] - z[:-2]), 1, 'edge') / frame.dt

    return frame


def parse_files(folder):

    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(folder):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    frames = []
    for f in tqdm(listOfFiles):
        if f[-3:].lower() == 'gpx':
            try:
                frames.append(parse_honzo(f))
            except ET.ParseError:
                print(f)
            except RuntimeError as re:
                print(re)
            except AttributeError as ae:
                print(ae, f)
            except ValueError as ve:
                print(ve, f)

    # Add UTM projection coordinates
    for df in frames:
        zone = get_utm_zone(df.lon.iloc[0])
        myProj = Proj(proj='utm', ellps='WGS84', zone=zone, units='m')

        x, y = myProj(df.lon.to_numpy(), df.lat.to_numpy())

        df['x'] = x
        df['y'] = y



    # dataFrame = pd.concat(frames, ignore_index=True)

    return frames



def segment_thermals(dataFrame, sigma=10):
    """
    Assuming the data frame contains one day
    :param dataFrame:
    :return:
    """

    from sklearn.cluster import DBSCAN
    dataFrame.vario = gaussian_filter1d(dataFrame['vario_est'].values, sigma)
    positiveVario = dataFrame[dataFrame.vario >= 0]

    if not positiveVario.size:
        positiveVario['labels'] = pd.Series([], dtype='int')
        return positiveVario

    fit = pd.DataFrame(positiveVario[['x', 'y', 'alt']])
    fit['t'] = positiveVario.time.values.astype(float) * 1e-9
    fit['t'] = fit['t'] * 0.5
    clustering = DBSCAN(eps=20, min_samples=3).fit(fit)
    positiveVario = pd.DataFrame(positiveVario)
    positiveVario['labels'] = positiveVario.flight_id + clustering.labels_

    return positiveVario


def cluster_thermals(frames):
    thermals = []
    for frame in tqdm(frames):
        seg = segment_thermals(frame, sigma=10)

        for l in seg.labels.unique():
            s = seg[seg.labels == l]
            if s.altitude.max() - s.altitude.min() > 100:
                thermals.append(pd.DataFrame(s))
    return thermals

if __name__ == '__main__':

    # ds = parse_honzo('../../data/Vario/VariYo.gpx')
    # ds.to_csv('VariYo.csv')
    #

    folder = '../../data/Honzo/19-2/19/'
    f_list = os.listdir(folder)

    if os.path.isfile('honzo_data.csv'):
        dataFrame = pd.read_csv('honzo_data.csv', parse_dates=['time'])

        frames = []
        for f in dataFrame.flight_id.unique():
            frames.append(dataFrame[dataFrame.flight_id == f])

    else:
        frames = parse_files(folder)
        dataFrame = pd.concat(frames, ignore_index=True)
        dataFrame.to_csv('honzo_data.csv')

    if os.path.isfile('honzo_thermals.csv'):
        df = pd.read_csv('honzo_thermals.csv', parse_dates=['time'])

        thermals = []
        for f in df.labels.unique():
            thermals.append(df[df.labels == f])
    else:
        thermals = cluster_thermals(frames)
        print(len(thermals), 'thermals produced')
        df = pd.concat(thermals)
        df.to_csv('honzo_thermals.csv')

    # plt.plot(df.time, df.altitude)
    plt.plot(df.time, df.vario)
    plt.show()

    for s in thermals:
        plt.plot(s.x, s.y, label=str(s.iloc[0].labels))
    plt.legend()
    plt.show()

