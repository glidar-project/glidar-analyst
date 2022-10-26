import matplotlib
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy, QStyle

from glidar_analyst.gui.matplot_vidgets import MplWidget3D, MplWidget
from glidar_analyst.gui.my_base_widget import MyBaseWidget

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets


class ThermalThumbnail(MyBaseWidget):

    thumbnailSelected = pyqtSignal(object)
    rangeSelected = pyqtSignal(object)

    def __init__(self, label, data, parent=None):

        super(ThermalThumbnail, self).__init__(parent)

        pal = self.palette()
        pal.setColor(QPalette.Background, QtCore.Qt.magenta)
        self.setAutoFillBackground(True)
        self.setPalette(pal)

        # This is needed to enable any key
        # events in the first place
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.model = None

        self.label = label
        self.data = data
        self.mplWidget = MplWidget(parent=self)
        # self.mplWidget.selectionChanged.connect(self.on_range_select)

        self.mplWidget.plot(data['vario'], data['altitude'], 'bo')
        self.mplWidget.sc.axes.text(0.1, 10, f"{data.time.dt.date.iloc[0]}", fontsize=20)
        self.mplWidget.sc.axes.set_xlim(0, 10)
        self.mplWidget.sc.axes.set_ylim(0, 2000)
        self.mplWidget.sc.axes.set_xticklabels([])
        self.mplWidget.sc.axes.set_yticklabels([])
        self.mplWidget.sc.fig.tight_layout()

        cid = self.mplWidget.sc.mpl_connect('button_press_event', lambda pos: self.select())

        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.mplWidget)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.setLayout(layout)
        self.resize(self.sizeHint())

    def minimumSize(self) -> QtCore.QSize:
        return QtCore.QSize(120,200)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(120,200)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(120,200)

    def plot_model(self, tup):
        print('Updating model')
        w, z = tup[0], tup[1]
        if self.model is not None:
            self.model.pop().remove()
            self.model = None
        self.model = self.mplWidget.sc.axes.plot(w, z, 'k-')
        self.mplWidget.sc.draw()

    def on_range_select(self, idx):
        print('range selected', idx)
        self.select()
        self.rangeSelected.emit((self,idx))

    def deselect(self):
        print('Deselected:', self)
        self.mplWidget.sc.axes.set_facecolor('white')
        # self.mplWidget.sc.axes.spines['left'].set_color('green')
        self.mplWidget.sc.draw()
        # self.setStyleSheet('')
        self.repaint()

        # pal = self.mplWidget.palette()
        # pal.setColor(QPalette.Background, QtCore.Qt.blue)
        # self.mplWidget.setPalette(pal)

    def select(self):
        print('Selected:', self)
        self.mplWidget.sc.axes.set_facecolor('lightblue')
        self.mplWidget.sc.draw()
        # self.mplWidget.setStyleSheet('border: 2px solid black;')
        # self.setStyleSheet("background-color:black;")
        self.repaint()

        pal = self.mplWidget.palette()
        pal.setColor(QPalette.Background, QtCore.Qt.red)
        self.mplWidget.setPalette(pal)

        self.thumbnailSelected.emit(self)

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(a0)

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(a0)
        print('clicked on the widget label=', self.label)
        self.select()

    # def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
    #     print('my key press')
    #     # super().keyPressEvent(a0)
    #     # sc.mpl_connect('key_prThermalThumbnailess_event', self.toggle_selector)
    #     # self.toggle_selector(a0)
    #     if a0.key() in [QtCore.Qt.Key_Delete]:
    #         print('Deleting data points in range; ', self.mplWidget.RS.extents)


class ThermalsStrip(MyBaseWidget):

    thermal_selection_changed = pyqtSignal(object)
    points_selected = pyqtSignal(object)

    def create_tiles(self, df):
        l = df['labels'].value_counts()

        self.df = df
        self.labels = l.index[:100]
        self.selected_thermal = []

        print('Preparing Thumbnails...')
        self.thumbnails = [ThermalThumbnail(label, df[df['labels'] == label], parent=self) for label in self.labels]
        print('Done.')
        for t in self.thumbnails:
            t.thumbnailSelected.connect(self.select_thermal)
            t.rangeSelected.connect(self.select_range)
        self.place_thumbnails()

        if self.thumbnails is not None and len(self.thumbnails) > 0:
            scroll_pixel = self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
            self.scrollArea.setMinimumHeight(self.thumbnails[0].height() + scroll_pixel + 2)
            self.scrollArea.adjustSize()

        self.scrollArea.repaint()

    def __init__(self, *args, **kwargs):

        super(ThermalsStrip, self).__init__(*args, **kwargs)

        # pal = self.palette()
        # pal.setColor(QPalette.Background, QtCore.Qt.darkYellow)
        # self.setAutoFillBackground(True)
        # self.setPalette(pal)

        self.df = None
        self.labels = None
        self.thumbnails = None
        self.selected_thermal = []

        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.listWidget = QtWidgets.QFrame()
        # self.listWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.thumbnail_grid = QtWidgets.QHBoxLayout()
        self.thumbnail_grid.setContentsMargins(0,0,0,0)

        self.listWidget.setLayout(self.thumbnail_grid)
        self.scrollArea.setWidget(self.listWidget)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.scrollArea)

        self.setLayout(layout)

        # self.scrollArea.setMinimumHeight(200)
        # self.scrollArea.adjustSize()

    def place_thumbnails(self):
        print('Placing Thumbnails...')
        for idx, t in enumerate(self.thumbnails):
            self.thumbnail_grid.addWidget(t)
        print('Done.')

    def select_thermal(self, thm):

        if thm in self.selected_thermal:
            thm.deselect()
            self.selected_thermal.remove(thm)
            self.thermal_selection_changed.emit(self.selected_thermal)
            return

        for t in self.selected_thermal:
            t.deselect()
            self.selected_thermal = []

        self.selected_thermal.append(thm)
        self.thermal_selection_changed.emit(self.selected_thermal)

    def select_range(self, tup):

        print('Selecting range on', *tup)
        thm, idx = tup[0], tup[1]
        if thm in self.selected_thermal:
            self.points_selected.emit(idx)


class ThermalsView(QtWidgets.QWidget):

    def __init__(self, df, labels, model_control, *args, **kwargs):

        super(ThermalsView, self).__init__(*args, **kwargs)
        print('Creating thermals view')

        self.df = df
        self.labels = labels
        self.selected_thermal = None

        self.plot_3d = MplWidget3D()

        scrollArea = QtWidgets.QScrollArea()

        listWidget = QtWidgets.QWidget()
        self.thumbnail_grid = QtWidgets.QGridLayout()

        print('Preparing Thumbnails...')
        self.thumbnails = [ ThermalThumbnail(label, df[df['labels'] == label]) for label in self.labels ]
        print('Done.')
        for t in self.thumbnails:
            t.thumbnailSelected.connect(self.select_thermal)
            t.rangeSelected.connect(self.select_range)

            # Ignore the model updates for now
            # model_control.modelUpdated.connect(t.plot_model)

        self.place_thumbnails()

        listWidget.setLayout(self.thumbnail_grid)
        scrollArea.setWidget(listWidget)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.plot_3d)
        layout.addWidget(scrollArea)

        self.setLayout(layout)

    def place_thumbnails(self, n_cols=3):

        print('Placing Thumbnails...')
        for idx, t in enumerate(self.thumbnails):
            self.thumbnail_grid.addWidget(t, idx // n_cols, idx % n_cols)
        print('Done.')

    def select_thermal(self, thm):

        if thm is self.selected_thermal:
            return

        if self.selected_thermal is not None:
            self.selected_thermal.deselect()
        self.selected_thermal = thm

        dl = thm.data

        self.plot_3d.sc.axes.clear()
        self.plot_3d.plot(dl['x'], dl['y'], dl['altitude'], '.')

    def select_range(self, tup):
        print('Selecting range on', *tup)
        thm, idx = tup[0], tup[1]
        if self.selected_thermal is thm:
            self.plot_3d.select_indices(idx)
