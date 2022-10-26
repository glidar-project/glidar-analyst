import numpy as np
import os
import datetime

import pandas as pd


class IgcParser:

    PARSING_KEYS = {
        'pilot': ['HFPLTPILOT', 'HFPLTPILOTINCHARGE' ],
        'glider_type': ['HFGTYGLIDERTYPE', 'HPGTYGLIDERTYPE'],
        'glider_id': ['HFGIDGLIDERID', 'HPGIDGLIDERID'],
        'gps_datum': ['HFDTMGPSDATUM'],
        'vario_type': ['HFFTYFRTYPE'],
        'gps_receiver': ['HFGPSRECEIVER'],
        'time_zone': ['HFTZNTIMEZONE']
    }

    def __init__(self, filename):

        self.filename = filename
        self.takeoff_datetime = None
        self.track = None                     # pandas frame with time, latitude, longitude, and altitude
        self.attributes = None                # whatever else was parsed from the file

        self.parse(filename)

    def parse(self, file):
        """
        0 1           7              15              23  25      29
        B H H M M S S D D M M M M M N D D D M M M M M E V P P P P P CR LF
        Description     Size              Element          Remarks
        Time            6 bytes           HHMMSS           Valid characters 0-9
        Latitude        8 bytes           DDMMMMMN         Valid characters N, S, 0-9
        Longitude       9 bytes           DDDMMMMME        Valid characters E,W, 0-9
        Fix valid       1 byte            V                A: valid, V:nav warning
        Press Alt.      5 bytes           PPPPP            Valid characters -, 0-9

        :param file:
        :return: None
        """

        time_arr = []
        space_arr = []

        parsed = dict()

        with open(file, 'r') as f:
            time_line = None
            for line in f:

                for k, v in IgcParser.PARSING_KEYS.items():
                    chunks = line.strip().split(':')
                    if chunks[0] in v:
                        parsed[k] = chunks[1]

                if line[0:5] == 'HFDTE':
                    time_line = line

                if line[0] == 'B' and line[24] == 'A':
                    point_time = datetime.datetime.strptime(line[1:7], '%H%M%S').time()
                    point_lat = float(line[ 7: 9]) + float( line[ 9:11] + '.' + line[11:14])/60.0
                    point_lon = float(line[15:18]) + float( line[18:20] + '.' + line[20:23])/60.0
                    point_alt = float(line[25:30])
                    time_arr.append(point_time)
                    space_arr.append([point_lon, point_lat, point_alt])

        self.attributes = parsed

        d = self.parse_date(time_line)
        takeoff_time = time_arr[0]
        self.takeoff_datetime = datetime.datetime.combine(d, takeoff_time)

        time = [ datetime.datetime.combine(d, t) for t in time_arr ]
        track = np.array(space_arr, float)

        self.track = pd.DataFrame(zip(time, *track.T), columns=['time', 'longitude', 'latitude', 'altitude'] )

    @staticmethod
    def parse_date(time_line):
        """
        Collected possibilities:
            HFDTE310520
            HFDTEDATE:290418,01

        :param time_line:
        :return: parsed datetime.date object
        """
        t = list(filter(str.isdigit, time_line))

        day = int(t[0] + t[1])      # Not a bug, string concatenation instead of int addition...
        month = int(t[2] + t[3])
        year = int(t[4] + t[5])

        if year >= 70:
            year += 1900
        else:
            year += 2000

        d = datetime.date(year, month, day)
        return d


if __name__ == '__main__':

    kk = 'HFDTEDATE:290418,01'
    d = IgcParser.parse_date(kk)
    print(d)

    # file = "../data/OLC/2020-05-31--Bomoen/05vxx941.igc"
    file = "../../data/IGC/2018-04-29-Voss/2018-04-29_10_37_29--Dale--Broad.igc"

    p = IgcParser(file)
    print(p.track)
