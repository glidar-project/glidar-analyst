import os
import datetime

from PyQt5.QtWidgets import QApplication, QFileDialog

from glidar_analyst.para.data_objects import Track, Glider, Pilot
from glidar_analyst.para.db_manager import DBManager
from glidar_analyst.para.igc_parser import IgcParser

file = "../../data/IGC/2018-04-29-Voss/2018-04-29_10_37_29--Dale--Broad.igc"


def add_track_to_db(filenname):
    print('\tParsing file:', filenname)

    parser = IgcParser(filenname)

    db = DBManager()

    pilot_name = parser.attributes['pilot']
    pilot = db.find_pilot_by_name(pilot_name)

    if pilot is None:
        pilot = Pilot.new(pilot_name)
        pilot = db.create_pilot(pilot)

    glider_name = parser.attributes['glider_type']
    g = db.find_glider_by_pilot_and_name(pilot, glider_name)
    if g is None:
        g = Glider.new(pilot, glider_name, -1.5)
        g = db.create_glider(g)

    t = Track.new(pilot, g, parser.takeoff_datetime, parser.track)
    t = db.create_track(t)

    return t


def add_files_from_folder(folder):

    file_list = os.listdir(folder)
    for file in file_list:
        if file[-4:].lower() == '.igc':
            add_track_to_db(os.path.join(folder, file))


def track_day(date):

    db = DBManager()

    res = db.find_tracks_by_date(date)
    print(res)
    print(len(res))


def main():

    import sys

    app = QApplication(sys.argv)

    dialog = QFileDialog()
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    dialog.setFileMode(QFileDialog.Directory)
    # dialog.setNameFilter('CSV (*.csv *.CSV)')
    # dialog.setDirectory('..')

    if dialog.exec():
        folders = dialog.selectedFiles()
        for f in folders:
            print("Parsing folder: ", f)
            add_files_from_folder(f)


if __name__ == '__main__':

    track_day(datetime.date.fromisoformat('2019-08-07'))