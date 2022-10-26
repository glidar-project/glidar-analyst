import pandas as pd


class Pilot:

    def __init__(self):

        self._id = None
        self.name = None

    @staticmethod
    def new(name):
        p = Pilot()
        p.name = name
        return p

    def to_mongo(self):

        return self.__dict__

    @staticmethod
    def from_mongo(val):

        if val is None:
            return None

        p = Pilot()
        p.__dict__ = val
        return p


class Glider:

    def __init__(self):

        self._id = None
        self.pilot_id = None

        self.name = None

        self.sink_rate = None

    @staticmethod
    def new(pilot, name, sink_rate):
        g = Glider()
        g.pilot_id = pilot._id
        g.name = name
        g.sink_rate = sink_rate
        return g

    def to_mongo(self):

        return self.__dict__

    @staticmethod
    def from_mongo(val):

        if val is None:
            return None

        g = Glider()
        g.__dict__ = val
        return g


class Track:

    def __init__(self):

        self._id = None

        self.pilot_id = None
        self.glider_id = None

        self.flight_date = None
        self.data_frame = None

    @staticmethod
    def new(pilot, glider, flight_date, data_frame):
        t = Track()
        t.pilot_id = pilot._id
        t.glider_id = glider._id
        t.flight_date = flight_date
        t.data_frame = data_frame
        return t

    def to_mongo(self):

        result = dict()
        result['pilot'] = self.pilot_id
        result['glider'] = self.glider_id
        result['flight_date'] = self.flight_date
        result['data_frame'] = self.data_frame.to_dict('list')
        return result

    @staticmethod
    def from_mongo(val):

        if val is None:
            return None

        track = Track()
        track.pilot_id = val['pilot']
        track.glider_id = val['glider']
        track.flight_date = val['flight_date']
        track.data_frame = pd.DataFrame(val['data_frame'])
        return track


class Thermal:

    def __init__(self, df):

        self.data_frame = df


class TrackAnalysis:

    def __init__(self):

        self.track = None

        self.utm_zone = None

        self.thermals = []
