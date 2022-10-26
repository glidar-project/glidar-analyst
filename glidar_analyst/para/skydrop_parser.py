import pandas as pd
import numpy as np

from scipy.ndimage import gaussian_filter1d

from datetime import datetime, date
import metpy.calc as mpcalc
from metpy.units import units
import ctypes

from PyQt5.QtWidgets import QApplication, QFileDialog

from glidar_analyst.para.igc_parser import IgcParser


class SkyDropIgcParser:

    def __init__(self):

        self.track = None                     # dataframe with columns: lat, lon, alt, time [datetime]
        self.attributes = None

    def parse_file(self, filename):

        flight_date = None
        data = []
        parsed = dict()

        with open(filename) as file:

            row = dict()

            for line in file:

                for k, v in IgcParser.PARSING_KEYS.items():
                    if line[0:len(v)] == v:
                        parsed[k] = line[len(v)+1:].strip()

                if line[0:5] == 'HFDTE':

                    flight_date = IgcParser.parse_date(line)

                if line[0] == 'B' and line[24] == 'A':
                    data.append(row)
                    row = dict()

                    point_time = datetime.strptime(line[1:7], '%H%M%S').time()
                    point_lat = float(line[7: 9]) + float(line[9:11] + '.' + line[11:14]) / 60.0
                    point_lon = float(line[15:18]) + float(line[18:20] + '.' + line[20:23]) / 60.0
                    point_alt = float(line[25:30])

                    row['time'] = point_time
                    row['longitude'] = point_lon
                    row['latitude'] = point_lat
                    row['altitude'] = point_alt

                if line[:10] == 'LXSB N8ExV':
                    row['temperature'] = float(line[10:16]) / 100.
                    row['RH'] = float(line[16:20]) / 100.
                    row['pressure'] = float(line[20:])

        if len(data) < 1:
            raise RuntimeError('No data found in file.', filename)

        if flight_date is None:
            raise RuntimeError('No date specified in the file', filename)

        data.pop(0)     # Get rid of the first empty row.
        df = pd.DataFrame(data)
        df['time'] = df['time'].apply(lambda t: datetime.combine(flight_date, t))
        df['dewpoint'] = mpcalc.dewpoint_from_relative_humidity(df.temperature.values * units.celsius,
                                                                df.RH.values * units.percent)
        self.track = df
        self.attributes = parsed
        self.attributes['date'] = flight_date
        flight_id = ctypes.c_size_t(file.__hash__()).value
        self.attributes['flight_id'] = flight_id


        # df['seconds'] = (df['time'] - df['time'][0]).apply(lambda d: d.seconds)
        # df['dt'] = np.pad((df.seconds.to_numpy()[2:] - df.seconds.to_numpy()[:-2]), 1, 'edge')
        # z = gaussian_filter1d(df.alt.values, 10)
        # df['vario_est'] = np.pad((z[2:] - z[:-2]), 1, 'edge') / df.dt
        #

        #
        # # Hacky part to get it to show in the tool...
        # df['labels'] = np.zeros_like(df.index)
        # df['vario'] = df['vario_est']
        # df['altitude'] = df.alt

        return df

    def to_mongo_dict(self):

        result = self.attributes.copy()
        result['date'] = datetime.combine(result['date'], datetime.min.time())

        for c in self.track.columns:
            result[c] = self.track[c].to_numpy().tolist()

        result['time'] = [ d.to_pydatetime() for d in self.track.time ]
        return result


def load_to_mongo(filename):

    import pymongo

    client = pymongo.MongoClient('localhost', 27017)
    database = client["test-tracking-data"]
    tracklogs = database["tracklogs"]

    parser = SkyDropIgcParser()
    parser.parse_file(filename)

    tracklogs.insert_one(parser.to_mongo_dict())

    print('done')


if __name__ == '__main__':

    import sys
    app = QApplication(sys.argv)

    dialog = QFileDialog()
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilter('IGC (*.igc *.IGC)')

    if (dialog.exec()):
        fname = dialog.selectedFiles()[0]
        print("Parsing file: ", fname)

        parser = SkyDropIgcParser()
        dataset = parser.parse_file(fname)
        dataset.to_csv(fname[:-4] + '_parsed.csv')

        # load_to_mongo(fname)


