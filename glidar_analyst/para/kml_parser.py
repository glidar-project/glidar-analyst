import numpy as np
import pandas as pd
import datetime
import xml.etree.ElementTree as ET


class KmlParser:

    def __init__(self, filename):
        self.filename = filename
        self.takeoff_datetime = None
        # self.time_sec = None
        self.track = None               # pandas dataframe w
        # self.time_step = None

        self.parse(filename)

    def parse(self, file):

        tree = ET.parse(file)
        root = tree.getroot()

        takeoff_time_text = root.find('Folder')[-1].find('Metadata').find('FsInfo').attrib['time_of_first_point']
        self.takeoff_datetime = datetime.datetime.strptime(takeoff_time_text[:18], "%Y-%m-%dT%H:%M:%S")

        time_sec = np.array(
            root.find('Folder')[-1]
                .find('Metadata')
                .find('FsInfo')
                .find('SecondsFromTimeOfFirstPoint')
                .text.split(),
            dtype=int)

        time = [ self.takeoff_datetime + datetime.timedelta(seconds=int(t)) for t in time_sec ]

        # self.time_step = np.median(self.time_sec[1:] - self.time_sec[:-1])

        track = np.array(
            [line.split(',') for line in root.find('Folder')[-1]
                .find('LineString')
                .find('coordinates')
                .text.split()],
            dtype=float)

        self.track = pd.DataFrame(zip(time, *track.T), columns=['time', 'longitude', 'latitude', 'altitude'] )


if __name__ == '__main__':
    test_filename = "../../data/Voss-2018-04-29/2018-04-29_Erik_Hamran_Nilsen.kml"
    # test_filename = "../../data/OLC/2020-05-30--Bomoen/05uv1fp1.kml"

    p = KmlParser(test_filename)
    print(p.track)


