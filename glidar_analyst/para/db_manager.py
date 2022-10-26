import pymongo

import datetime
from bson.objectid import ObjectId

from glidar_analyst.para.data_objects import Pilot, Glider, Track, Thermal


class DBManager:

    def __init__(self):

        self.client = pymongo.MongoClient('localhost', 27017)
        self.database = self.client['pg-db']

    #
    # Create Section
    def create_pilot(self, pilot: Pilot):

        pilot._id = ObjectId()
        self.database['pilots'].insert_one(pilot.to_mongo())
        return pilot

    def create_glider(self, glider: Glider):

        glider._id = ObjectId()
        self.database['gliders'].insert_one(glider.to_mongo())
        return glider

    def create_track(self, track: Track):

        track._id = ObjectId()
        self.database['tracks'].insert_one(track.to_mongo())
        return track

    #
    # Retrieve section
    def find_pilot_by_id(self, id):

        res = self.database.pilots.find_one({'_id': id})
        return Pilot.from_mongo(res)

    def find_pilot_by_name(self, name):

        res = self.database['pilots'].find_one({'name': name})
        return Pilot.from_mongo(res)

    def find_glider_by_id(self, id):

        res = self.database.gliders.find_one({'_id': id})
        return Glider.from_mongo(res)

    def find_glider_by_pilot_and_name(self, pilot, glider_name):

        res = self.database.gliders.find_one({'pilot_id': pilot._id, 'name': glider_name})
        return Glider.from_mongo(res)

    def find_track_by_id(self, id):

        res = self.database.tracks.find_one({'_id': id})
        return Track.from_mongo(res)

    def find_tracks_by_date(self, date):

        start = datetime.datetime.combine(date, datetime.time.min)
        end = datetime.datetime.combine(date, datetime.time.max)

        print(start)
        print(end)

        res = self.database.tracks.find({
            'flight_date': {'$gte': start, '$lt': end}
        })

        return [Track.from_mongo(t) for t in res]

    #
    # Update section
    def update_pilot(self, pilot: Pilot):

        self.database['pilots'].update(pilot.to_mongo())

    def update_glider(self, glider: Glider):

        self.database['gliders'].update(glider.to_mongo())

    def update_track(self, track: Track):

        self.database['tracks'].update(track.to_mongo())

    #
    # Delete section
    def delete_pilot(self, pilot: Pilot):
        self.database['pilots'].delete_one(pilot.to_mongo())

    def delete_glider(self, glider: Glider):
        self.database['gliders'].delete_one(glider.to_mongo())

    def delete_track(self, track: Track):
        self.database['tracks'].delete_one(track.to_mongo())
