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

import simplekml

from glidar_analyst.para.igc_parser import IgcParser
from glidar_analyst.para.kml_parser import KmlParser


def parse_files(folder, sigma=10):

    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(folder):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    flights = []
    for f in tqdm(listOfFiles):
        if f[-3:].lower() == 'gpx':
            flights.append(IgcParser(f))
        elif f[-3:].lower() == 'kml':
            flights.append(KmlParser(f))

    frames = []
    for flight in flights:
        print(flight.filename)
        df = pd.DataFrame(np.concatenate([flight.time_sec.reshape(flight.time_sec.size, 1 ), flight.track], axis=1),
                          columns=['time', 'longitude', 'latitude', 'altitude'])

        df['dt'] = np.pad((df.time.values[2:] - df.time.values[:-2]), 1, 'edge')
        z = gaussian_filter1d(df.altitude.values, sigma)
        df['vario'] = np.pad((z[2:] - z[:-2]), 1, 'edge') / df.dt

        myProj = Proj(proj='utm', ellps='WGS84', zone=32, units='m')
        x, y = myProj(df.longitude.to_numpy(), df.latitude.to_numpy())
        df['x'] = x
        df['y'] = y

        df = pd.DataFrame(df[gaussian_filter1d(df.vario.values, sigma) > 0])
        frames.append(df)

    return frames


def gps_to_ecef_custom(lat, lon, alt):
    '''
    https://gis.stackexchange.com/questions/230160/converting-wgs84-to-ecef-in-python
    :param lat:
    :param lon:
    :param alt:
    :return:
    '''
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)

    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))

    x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (v * (1 - e2) + alt) * np.sin(rad_lat)

    return x, y, z


if __name__ == '__main__':

    frames = parse_files('../../data/Flightlog')
    df = pd.concat(frames)
    df['labels'] = np.ones_like(df.x.values)
    df.to_csv('flightlog2019.csv')

    # import laspy
    #
    # with laspy.file.File('./House.laz', mode='r') as f:
    #     print(f.header)
    #     h = f.header
    #     print('############################################')
    #
    #     # Find out what the point format looks like.
    #     pointformat = f.point_format
    #     for spec in f.point_format:
    #         print(spec.name)
    #
    #     print('############################################')
    #     # Lets take a look at the header also.
    #     headerformat = f.header.header_format
    #     for spec in headerformat:
    #         print(spec.name)
    #
    #     # Like XML or etree objects instead?
    #     a_mess_of_xml = pointformat.xml()
    #     an_etree_object = pointformat.etree()
    #
    #     print(an_etree_object)
    #     # It looks like we have color data in this file, so we can grab:
    #     blue = f.blue
    #     print(f.X)
    #     print(f.x)
    #
    #     with laspy.file.File("./data.las", mode="w", header=h) as of:
    #         # f.define_new_dimension(name="X", data_type=5, description="My new dimension")
    #         # f.my_dim = ...
    #         # ...
    #         print('############################################')
    #         print()
    #
    #         print(of.header.offset)
    #         print(of.header.scale)
    #
    #         of.header.offset = (0,0,0)
    #         of.header.scale = (1,1,1)
    #
    #         print(of.header.offset)
    #         print(of.header.scale)
    #
    #         # x, y, z = gps_to_ecef_custom(df.latitude.values, df.longitude.values, df.altitude.values)
    #
    #         # of.header.offset = (-np.mean(x), -np.mean(y), -np.mean(z))
    #
    #         a = df[['x', 'y', 'altitude']].min()
    #         b = df[['x', 'y', 'altitude']].max()
    #         c = a + 0.5*(b - a)
    #         print(a)
    #         print(b)
    #         print(c)
    #
    #         of.X = df.x.values - c.x
    #         of.Y = df.y.values - c.y
    #         of.Z = df.altitude.values - c.altitude
    #
    #         print(df.x.mean(), df.y.mean(), df.altitude.mean())
    #
    #         myProj = Proj(proj='utm', ellps='WGS84', zone=32, units='m')
    #         lon, lat = myProj(c.x, c.y, inverse=True)
    #
    #         print('origin: ', lat, lon, c.altitude)
