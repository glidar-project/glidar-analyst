import sys

import matplotlib
from PyQt5.QtCore import QThreadPool
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QFileDialog, QDockWidget, QApplication

# from glidar_analyst.gui.thermal_3d_view import Thermal3DView
from glidar_analyst.para.data_loader import load_and_segment
from glidar_analyst.gui.model_fitting_widget import ModelFittingWidget
from glidar_analyst.util.resource_manager import ResourceManager
from glidar_analyst.worker import Worker

from PyQt5 import QtCore, QtGui, QtWidgets

import pandas as pd
import netCDF4 as nc
import xarray as xr

import os

matplotlib.use('Qt5Agg')


class MainWindow(QtWidgets.QMainWindow):
    #
    # Thread pool for loading data mostly
    # Guess this should be class level, not object level
    threadpool = QThreadPool()

    def __init__(self, *args, **kwargs):

        #
        # Super-Constructor
        super(MainWindow, self).__init__(*args, **kwargs)

        #
        # Setting up the menu
        self.setUpMenuBar()

        self.thumbnails_view = None

        self.model_fitting_view = ModelFittingWidget(parent=self)
        self.setCentralWidget(self.model_fitting_view)

        self.model_control = self.model_fitting_view.model_controller
        #
        # Floating model control widget
        dockWidget = QDockWidget('Model Controls', self)
        dockWidget.setWidget(self.model_control)
        dockWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea
                                   | QtCore.Qt.RightDockWidgetArea)
        dockWidget.setFeatures(QDockWidget.DockWidgetFloatable
                               | QDockWidget.DockWidgetMovable)
        dockWidget.setFloating(False)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dockWidget)

        #
        # Init the thing with default dataset
        filename = ResourceManager().get_absolute_path(
            '../data/2022-Voss_imet_soundings.nc')
        # filename = ResourceManager().get_absolute_path(
        #     '../data/Voss-2018-04-29/sounding/' + 'sola_20180331-20180430.nc')

        try:
            self.sounding_loaded(
                (nc.Dataset(filename), xr.open_dataset(filename).to_dataframe()))
            
            self.data_loaded(
                pd.read_csv(ResourceManager().get_absolute_path(
                    '../data/Voss_iMet_thermals/20220516-234233-00044494_thermals.csv'),
                            parse_dates=['time']))
            self.data_loaded(
                pd.read_csv(ResourceManager().get_absolute_path(
                    '../data/Voss_iMet_thermals/20220517-184531-00044494_thermals.csv'),
                            parse_dates=['time']))
        except FileNotFoundError as e:
            print('Data not found. Download the data from https://github.com/glidar-project/glidar-analyst/releases/download/0.1.0/data.zip and put them in the root folder.')

        # self.data_loaded(pd.read_csv(ResourceManager().get_absolute_path('../data/Voss-vol2/clusters.csv'), parse_dates=['time']))
        # self.data_loaded(pd.read_csv('../data/OLC/olc_thermals.csv', parse_dates=['time']))
        # self.data_loaded(pd.read_csv('olc_frame.csv', parse_dates=['time']))

        self.show()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        QApplication.quit()

    def open_sounding_data(self):

        dialog = QFileDialog()
        dialog.setNameFilter('NC (*.nc *.NC)')
        dialog.setDirectory(ResourceManager().get_absolute_path(
            '../data/Voss-2018-04-29/sounding/'))
        dialog.setFileMode(QFileDialog.ExistingFile)

        if (dialog.exec()):
            fname = dialog.selectedFiles()[0]

            def worker_function(filename):
                nc_data = nc.Dataset(filename)
                xarr = xr.open_dataset(filename)
                return nc_data, xarr.to_dataframe()

            w = Worker(worker_function, fname)
            w.signals.result.connect(self.sounding_loaded)
            self.threadpool.start(w)

    def sounding_loaded(self, args):
        data = args[0]
        xarr = args[1]
        # print(data)
        # print(xarr)
        self.model_fitting_view.set_sounding_data(data, xarr)

    def openFileDialog(self):
        dialog = QFileDialog()
        dialog.setNameFilter('CSV (*.csv *.CSV)')
        dialog.setDirectory('..')
        dialog.setFileMode(QFileDialog.ExistingFile)

        if (dialog.exec()):
            fname = dialog.selectedFiles()[0]
            w = Worker(lambda file: pd.read_csv(file, parse_dates=['time']),
                       fname)
            w.signals.result.connect(self.data_loaded)
            self.threadpool.start(w)

    def loadDataDialog(self):
        dialog = QFileDialog()
        # dialog.setNameFilter()
        dialog.setDirectory('..')
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly)

        if (dialog.exec()):
            fname = dialog.selectedFiles()[0]
            import os
            print(fname)
            if os.path.isfile(fname):
                fname = os.path.dirname(fname)
                print(fname)
            self.load_data(fname)

    def load_data(self, folder) -> None:
        w = Worker(load_and_segment, folder)
        w.signals.result.connect(self.data_loaded)
        # TODO: Progressbar
        self.threadpool.start(w)

    def data_loaded(self, data_frame) -> None:
        # self.datasets.append(data_frame)
        print('loading new dataset:', data_frame)

        l = data_frame['labels'].value_counts()
        if self.model_fitting_view is not None:
            self.model_fitting_view.set_thermal_data(data_frame)

        # self.thumbnails_view = ThermalsView(data_frame, l.index[:100])
        # self.setCentralWidget(self.thumbnails_view)
        self.repaint()

    def setUpMenuBar(self):
        #
        # File menu
        fm = self.menuBar().addMenu('File')

        #
        # Open saved data frame file from csv
        openFile = QAction('Load segmented thermals', fm)
        openFile.triggered.connect(self.openFileDialog)
        fm.addAction(openFile)

        # #
        # # Load kml tracklog data from a folder
        # loadData = QAction('Parse tracklog files', fm)
        # loadData.triggered.connect(self.loadDataDialog)
        # fm.addAction(loadData)

        #
        # Load netCDF4 sounding data
        loadSounding = QAction('Load Sounding', fm)
        loadSounding.triggered.connect(self.open_sounding_data)
        fm.addAction(loadSounding)


def main():

    DIRNAME = os.path.dirname(os.path.abspath(__file__))
    ResourceManager(DIRNAME)
    print(ResourceManager().base_path)

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.setWindowTitle('gLidar Analyst')
    w.setWindowIcon(
        QIcon(ResourceManager().get_absolute_path('../icon/icon.png')))

    # thermal = Thermal3DView(w)
    # thermal.show()
    # w.model_fitting_view.thermal3d = thermal

    app.exec_()


if __name__ == '__main__':

    main()
