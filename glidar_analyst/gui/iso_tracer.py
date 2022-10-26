import time
import typing

import matplotlib.pyplot as plt

import matplotlib
from PyQt5.QtCore import pyqtSignal, Qt, QPoint, QAbstractTableModel, QModelIndex, QItemSelectionModel, QTimer
from PyQt5.QtGui import QColor, QBrush, QIcon
from PyQt5.QtWidgets import QMenu, QAction, QHBoxLayout, QVBoxLayout, \
    QApplication, QSizePolicy, QPushButton, QWidget, \
    QTableView, QAbstractItemView, QStyledItemDelegate, QStyle, QSplitter
from numpy.linalg import LinAlgError
from pyqtgraph import GraphicsLayoutWidget, ScatterPlotItem, PlotItem

from glidar_analyst.iso_trotter import TracerParamObject, IsoTrotter, Target, TracerObjectDatabase
from glidar_analyst.gui.matplot_vidgets import MplWidget
from glidar_analyst.model.simple_model import QuadraticModel
from glidar_analyst.gui.model_controls_widget import RealModelControlsWidget
from glidar_analyst.util.resource_manager import ResourceManager

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pandas as pd

from pint import UnitRegistry
from scipy.stats import ortho_group

units = UnitRegistry()


class ParamColorPallete:

    def __init__(self, params):

        self.params = params
        self.colors = plt.get_cmap('tab10')

    def get_color(self, p):

        i = self.params.index(p)
        return self.colors((i % 10) / 9)


class ClicFitTarget:

    def __init__(self, x, y, var, parent):

        self.parent = parent
        self.target_x = x
        self.target_y = y
        self.variable = var

        self.mpl_object = None

        self.redraw()

        self.is_green = False
        self.timer = None
        self.flash_count = 3

    def __del__(self):

        if self.mpl_object is not None:
            for e in self.mpl_object:
                e.remove()

        if self.parent and self.parent.mpl_view and self.parent.mpl_view.sc:
            self.parent.mpl_view.sc.draw()

        print('Destroyed.')

    def update(self, frame):
        for e in self.mpl_object:
            e.set_visible(not e.get_visible())
        return self.mpl_object

    def animate(self):

        print('Flashing.')

        self.timer = QTimer()
        self.timer.timeout.connect(self.flesh)
        self.timer.start(300)

    def hide(self):

        if self.mpl_object is not None:
            for e in self.mpl_object:
                e.remove()
            self.mpl_object = None

    def confirm(self):

        if self.mpl_object is not None:
            for e in self.mpl_object:
                e.set_color('red')

    def flesh(self):

        self.flash_count -= 1
        if self.flash_count < 0:
            self.timer.stop()
            self.flash_count = 3
            # self.hide()
            self.parent.flashing_targets.remove(self)

            if self.parent:
                self.parent.mpl_view.sc.draw()
            return

        if self.mpl_object is not None:
            for e in self.mpl_object:
                if self.is_green:
                    e.set_color('red')
                else:
                    e.set_color('green')
                self.is_green = not self.is_green

                self.parent.mpl_view.sc.draw()

    def redraw(self):
        print('Drawing dot', self.target_x, self.target_y, self.parent.color_mapper.get_color(self.variable))
        if self.mpl_object is not None:
            for e in self.mpl_object:
                e.remove()
            self.mpl_object = None

        self.mpl_object = self.parent.mpl_view.sc.axes.plot(self.target_y, self.target_x, 'x',
                                                            markersize=20,
                                                            markeredgewidth=5,
                                                            c='green'
                                                            # c=self.parent.color_mapper.get_color(self.variable)
                                                            )

    def to_target(self):

        return Target(self.target_x, self.target_y, self.variable)


class ClickFitter:

    def __init__(self, parent, model, mpl_view):

        self.color_mapper = ParamColorPallete(model.params)
        self.parent = parent
        self.model = model
        # self.solver = BisectionSolver(model)
        # self.solver = NewtonSolver(model)
        self.mpl_view = mpl_view

        self.targets = dict()
        self.flashing_targets = []

        ##################################################
        # Click Fitting Menu Setup
        self.click_fit_menu = QMenu('Select parameter to fit', self.mpl_view)
        names = self.model.params
        if hasattr(self.model, 'names'):
            names = self.model.names.values()
        self.click_fit_actions = {QAction(n, self.parent): p for (p, n) in zip(self.model.params, names)}
        for a in self.click_fit_actions.keys():
            self.click_fit_menu.addAction(a)
        cancel = QAction("Cancel Anchoring", self.parent)
        self.click_fit_actions[cancel] = None
        self.click_fit_menu.addAction(cancel)

        self.mpl_view.sc.mpl_connect('button_press_event', self.click_fitting)
        ##################################################

    def get_targets(self):
        """
        Translates the ClickFitTarget objects to the dummy data transfer objects.
        :return:
        """
        return { k: v.to_target() for k, v in self.targets.items() }

    def paint_result(self, result: TracerParamObject):
        """
        After the model is done computing this thing needs
        to show the active anchor points.
        Right now we will draw directly into the mpl view attribute,
        this might change in the future.
        :param result:
        :return:
        """
        targets = result.targets

        self.targets.clear()
        for t in targets.values():
            self.targets[t.variable] = ClicFitTarget(t.target_x, t.target_y, t.variable, self)

        self.mpl_view.sc.draw()

    def confirm_anchor(self, variable):

        if variable in self.targets.keys():
            self.targets[variable].confirm()
        self.mpl_view.sc.draw()

    def flash_targets(self):

        for t in self.targets.values():
            self.flashing_targets += [t]
            t.animate()
        self.mpl_view.sc.draw()

    def cancell_anchor(self, variable):

        if variable in self.targets.keys():
            self.flashing_targets += [self.targets[variable]]
            self.targets[variable].animate()
            self.targets[variable] = None
            del self.targets[variable]

        self.mpl_view.sc.draw()

    def click_fitting(self, event):

        if self.model is None:
            return

        if event.button == 3:  # Right mouse button

            action = self.click_fit_menu.exec(self.mpl_view.mapToGlobal(
                                                QPoint(event.x, self.mpl_view.height() - event.y)))

            if not action:
                return

            target_y = event.xdata
            target_x = event.ydata
            click_fit_param = self.click_fit_actions[action]

            if click_fit_param is None:
                self.targets.clear()
                self.mpl_view.sc.draw()

            if click_fit_param:
                t = ClicFitTarget(target_x, target_y, click_fit_param, self)
                if click_fit_param in self.targets:
                    del self.targets[click_fit_param]
                self.targets[click_fit_param] = t

                self.mpl_view.sc.draw()
                #
                # try:
                #     params = self.solver.solve_multi_target(self.parent.get_params(), self.get_targets())
                #     self.parent.set_params(params)
                #
                # except RuntimeError as re:
                #     del self.targets[click_fit_param]
                #     print('Could not solve the new set of constraints.')
                #
                # except LinAlgError as le:
                #     del self.targets[click_fit_param]
                #     print('Could not solve the new set of constraints.')

            # if hasattr(self.parent, 'click_fitting_targets'):
            self.parent.click_fitting_targets(self.get_targets(), click_fit_param)


class ParamTableData(QAbstractTableModel):

    def __init__(self, model):

        super(ParamTableData, self).__init__()

        self.model = model
        self.names = list(self.model.names.values()) + ['ID']
        self._data = pd.DataFrame(columns=list(self.model.params) + ['id']).astype({'id': int})
        self._data_view = self._data
        self._index_to_params = {}

        self.sort_column = -1
        self.sort_order = Qt.AscendingOrder

    def removeRows(self, row: int, count: int, parent: QModelIndex = None) -> bool:

        self.beginRemoveRows(parent, row, row + count - 1)
        selected = self._data_view.iloc[row : row + count]
        print(selected.index)
        result = True
        try:
            self._data_view.drop(selected.index, inplace=True)
            self._data.drop(selected.index, inplace=True)
        except Exception as e:
            print(e)
            result = False

        self.endRemoveRows()
        return result

    def delete_selected(self, selected):

        rows = np.array([e.row() for e in selected])
        rows = np.sort(rows)

        for r in rows[::-1]:
            self.removeRow(r)

    def get_sorted_selection(self, selection):

        rows = list([ i.row() for i in selection ])
        view = self._data_view.iloc[rows]
        ids = view.index
        val = view.iloc[:, self.sort_column]    # selects the sorted column or the ids

        return ids.values, val.values

    def get_index_by_id(self, id):
        # return self._data_view[self._data_view.id == id].index[0]
        # return Index(self._data_view.id).get_loc(id)
        return self._data_view.index.get_loc(id)

    def get_selected_row(self, index):
        return int(self._data_view.iloc[index.row()].id)

    def sort(self, column: int, order: Qt.SortOrder = ...) -> None:
        """
        https://stackoverflow.com/questions/42028534/pyqt-how-to-reimplement-qabstracttablemodel-sorting

        :param column:
        :param order:
        :return: None
        """
        self.sort_column = column
        self.sort_order = order

        self.layoutAboutToBeChanged.emit([])
        oldIndexList = self.persistentIndexList()
        oldIds = self._data_view.index.copy()

        if column == -1:
            self._data_view = self._data
        else:
            self._data_view = self._data.sort_values(self._data.columns[column],
                                                     0,
                                                     ascending=order == Qt.AscendingOrder)

        newIds = self._data_view.index
        newIndexList = []
        for index in oldIndexList:
            id = oldIds[index.row()]
            newRow = newIds.get_loc(id)
            newIndexList.append(self.index(newRow, index.column(), index.parent()))

        self.changePersistentIndexList(oldIndexList, newIndexList)
        self.layoutChanged.emit([])
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        return flags

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:

        assert self.checkIndex(index)

        if role == Qt.DisplayRole:
            return "{:.3e}".format(self._data_view.iloc[index.row(), index.column()])

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> typing.Any:

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section < self._data_view.shape[1]:
                return self.names[section]

        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return section + 1

        return None

    def rowCount(self, parent: QModelIndex = ...) -> int:

        if parent.isValid():
            return 0
        return self._data_view.shape[0]

    def columnCount(self, parent: QModelIndex = ...) -> int:

        if parent.isValid():
            return 0
        return self._data_view.shape[1]

    def add_entry(self, tracer_obj):

        row = tracer_obj.params.copy()
        row['id'] = tracer_obj.id

        self.beginInsertRows(QModelIndex(), self._data_view.shape[0], self._data_view.shape[0])

        self._data.loc[int(tracer_obj.id)] = row.values()

        self.endInsertRows()

        self.sort(self.sort_column, self.sort_order)


class RowHoverDelegate(QStyledItemDelegate):

    def __init__(self, *args, **kwargs):
        super(RowHoverDelegate, self).__init__(*args, **kwargs)

        self.hovered_row = -1

    def onHoverIndexChanged(self, item):

        self.hovered_row = item.row()

    def onLeaveTableEvent(self):

        self.hovered_row = -1

    def paint(self, painter: QtGui.QPainter, option: 'QStyleOptionViewItem', index: QtCore.QModelIndex) -> None:

        if index.row() == self.hovered_row:
            # option.state |= QStyle.State_MouseOver
            brush = QBrush(QColor(237, 164, 159))
            # // FILL BACKGROUND
            painter.save()
            painter.fillRect(option.rect, brush)
            painter.restore()
            option.state &= ~QStyle.State_Selected
        else:
            row_id = self.parent().param_table_data.get_selected_row(index)
            color = self.parent().selection_color_mapper.get_color(row_id)
            if color is not None:
                brush = QBrush(color)
                # // FILL BACKGROUND
                painter.save()
                painter.fillRect(option.rect, brush)
                painter.restore()
                option.state &= ~QStyle.State_Selected

        super(RowHoverDelegate, self).paint(painter, option, index)


class CustomTableView(QTableView):

    hoverIndexChanged = pyqtSignal(object)
    leaveTableSignal = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(CustomTableView, self).__init__(*args, **kwargs)

        self.hovered_index = None

        self.setMouseTracking(True)

        row_hover_delegate = RowHoverDelegate(self.parent())

        self.hoverIndexChanged.connect(row_hover_delegate.onHoverIndexChanged)
        self.leaveTableSignal.connect(row_hover_delegate.onLeaveTableEvent)

        self.setItemDelegate(row_hover_delegate)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        index = self.indexAt(e.pos())
        if self.hovered_index != index:
            self.hovered_index = index
            self.hoverIndexChanged.emit(index)
            self.viewport().update()

    def leaveEvent(self, a0: QtCore.QEvent) -> None:

        self.leaveTableSignal.emit()
        self.viewport().update()


class PandasTablePlot:

    symbol_size = 10
    hover_size = 15

    point_brush = QColor('white')
    active_point_brush = QColor('red')
    hover_point_pen = QColor('red')

    def __init__(self, data, parent, view, x_variable, y_variable, model_controls=None):

        self.data = data        # pandas table
        self.parent = parent
        self.view = view
        self.x_variable = x_variable
        self.y_variable = y_variable
        self.model_controls = model_controls

        self.view.clear()
        self.view.disableAutoRange()
        self.set_ranges()

        self.plot_data_item = ScatterPlotItem([],
                                              size=self.symbol_size,
                                              symbol='o',
                                              hoverable=True,
                                              brush=self.point_brush)

        self.plot_data_item.sigClicked.connect(self.point_clicked)
        self.plot_data_item.sigHovered.connect(self.point_hovered)

        self.view.addItem(self.plot_data_item)

    def set_ranges(self):
        if self.model_controls:
            self.view.setXRange(*self.model_controls.get_ranges()[self.x_variable], padding=0.1)
            self.view.setYRange(*self.model_controls.get_ranges()[self.y_variable], padding=0.1)

    def point_clicked(self, points, items):

        # print('Click event:', points, items)
        ids = [ i.data() for i in items ]
        self.parent.on_plot_click(ids)

    def point_hovered(self, points, items):

        # print('Point event:', points, items)
        ids = [ i.data() for i in items ]
        self.parent.on_plot_hover(ids)

    def update_plot(self):

        # print("Reading pandas", self.data.columns, self.x_variable)

        self.plot_data_item.setData(
            self.data[self.x_variable].to_numpy(),
            self.data[self.y_variable].to_numpy(),
            data=self.data.index
        )
        # self.view.autoRange()
        # self.set_ranges()

    def duplicate_plot(self, other_plot):

        self.data = other_plot.data
        self.x_variable = other_plot.x_variable
        self.y_variable = other_plot.y_variable

        self.view.clear()
        self.plot_data_item = ScatterPlotItem([],
                                              size=self.symbol_size,
                                              symbol='o',
                                              hoverable=True,
                                              brush=self.point_brush)

        self.update_plot()

    def _ids_to_idx(self, ids):
        #
        # if not isinstance(ids, list):
        #     ids = [ids]

        s = pd.Series(np.arange(self.data.shape[0]), index=self.data.index)
        idx = s.loc[list(ids)]
        return list(idx.values.flatten())

    def hover_id(self, ids):

        size = self.symbol_size * np.ones(self.data.shape[0])

        size[self._ids_to_idx(ids)] = self.hover_size

        self.plot_data_item.setSize(size)

    def highlight_id(self, obj_ids):
        """
        This one is used for showing the selection from the table view.
        :param obj_ids:
        :return:
        """
        c = [ self.point_brush for i in range(self.data.shape[0]) ]

        if len(obj_ids) > 0:
            colors = { i: QBrush(self.parent.selection_color_mapper.get_color(i)) for i in obj_ids }

            for i, j in zip(self._ids_to_idx(obj_ids), obj_ids):
                c[i] = colors[j]

        self.plot_data_item.setBrush(c)


class TracerPlot:

    symbol_size = 10
    hover_size = 15

    point_brush = QColor('white')
    active_point_brush = QColor('red')
    hover_point_pen = QColor('red')

    cmap = plt.get_cmap('Reds')

    def __init__(self, parent, view, x_variable, y_variable):

        self.parent = parent
        self.view = view
        self.x_variable = x_variable
        self.y_variable = y_variable

        self.view.clear()

        self.points = dict()
        self.active_points = []
        self.hovered = []

        self.view.disableAutoRange()


    # def point_hovered(self, points, event):
    #
    #     print('Point event:', points, event)

    def paint_point(self, tracer_object: TracerParamObject):

        _id = tracer_object.id
        point = tracer_object.params

        p = self.view.plot([point[self.x_variable]],
                           [point[self.y_variable]],
                           symbolSize=self.symbol_size,
                           symbol='o',
                           hoverable=False,
                           symbolBrush=self.point_brush,
                           data=_id)

        # The hover signal does not get fired...
        # p.sigPointsHovered.connect(self.point_hovered)
        # p.sigPointsClicked.connect(self.point_hovered)
        self.points[_id] = p
        # self.view.autoRange()

    def paint_points(self, points):

        for p in points:
            self.paint_point(p)

    def duplicate_plot(self, plot):

        self.points.clear()
        # Copying points into the big plot manually...
        for k, item in plot.points.items():
            print('Item data:', item.xData, item.yData)
            self.points[k] = self.view.plot(item.xData, item.yData, **item.opts)

        self.active_points.clear()

        for item in plot.active_points:
            print('Item data:', item.xData, item.yData)
            self.active_points.append(self.view.plot(item.xData, item.yData, **item.opts))

    def hover_id(self, ids):

        for e in self.hovered:
            # self.view.removeItem(e)
            e.setSymbolSize(self.symbol_size)

        self.hovered = []

        for i in ids:
            e = self.points[i]
            self.hovered.append(e)
            e.setSymbolSize(self.hover_size)

    def highlight_id(self, obj_ids):
        """
        This one is used for showing the selection from the table view.
        :param obj_ids:
        :return:
        """
        for p in self.active_points:
            # self.view.removeItem(p)
            p.setSymbolBrush(self.point_brush)

        self.active_points = []

        for i, pid in enumerate(obj_ids):
            c = self.parent.selection_color_mapper.get_color(pid)
            brush = QBrush(c)
            p = self.points[pid]
            p.setSymbolBrush(brush)
            self.active_points.append(p)


class SelectionColorMapper:

    cmap = plt.get_cmap('Reds')
    MIN = 0.3
    MAX = 0.7

    def __init__(self, ids, values):

        self.ids = ids
        self.values = values
        self.colors = dict()

        if len(ids) < 1:
            return

        self.min = np.min(values)
        self.max = np.max(values)

        self.linearized = values
        diff = (self.max - self.min)

        if len(self.linearized) > 1 and diff != 0:
            self.linearized = (self.linearized - self.min) / diff
        elif len(self.linearized) == 1:
            self.linearized = np.ones_like(ids)

        for k, v in zip(self.ids, self.linearized):
            mpl_c = self.cmap(self.remap(v))
            c = list([ int(c * 256) for c in mpl_c[:3] ])
            self.colors[k] = QColor(*c), mpl_c

    def remap(self, val):
        return self.MIN + val * (self.MAX - self.MIN)

    def get_color(self, id):
        """
        Finds the assigned color for the given index,
        returns None if index not selected.
        :param id:
        :return:
        """
        if id in self.colors.keys():
            return self.colors[id][0]
        return None

    def get_mpl_color(self, id):
        if id in self.colors.keys():
            return self.colors[id][1]
        return None


class IsoTracer(QtWidgets.QWidget):
    """
    The QtWidget that shows the scatterplot matrix and the table view.
    """

    param_points_hovered = pyqtSignal(object)
    tracer_point_selected = pyqtSignal(object)
    tracer_points_selected = pyqtSignal(object, object)

    def __init__(self, model, model_controls=None):

        super(IsoTracer, self).__init__()

        self.setWindowTitle('gLidar Analyst')
        self.setWindowIcon(QIcon(ResourceManager().get_absolute_path('../icon/icon.png')))

        self.model = model
        self.model_controls = model_controls
        # self.tracer_objects = {}

        self.setLayout(QHBoxLayout(self))

        self.splitter = QSplitter(self)

        self.selection_color_mapper = SelectionColorMapper([], [])

        ################################################################
        # Table View
        # self.param_table = QTableView(self)
        self.param_table = CustomTableView(self)
        # delegate = RowHoverDelegate()
        # self.param_table.setItemDelegate(delegate)
        self.param_table.setSortingEnabled(True)
        self.param_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.param_table_data = ParamTableData(self.model)
        self.param_table.setModel(self.param_table_data)

        self.param_table.clicked.connect(self.row_in_table_selected)
        self.param_table.selectionModel().selectionChanged.connect(self.table_selection_changed)

        ################################################################
        # Scatterplot Matrix
        self.scatterplot_matrix = GraphicsLayoutWidget()
        self.grid = dict()
        self.plot_names = dict()
        for i, pi in enumerate(model.params):
            for j, pj in enumerate(model.params):
                if i >= j:
                    w = self.scatterplot_matrix.addPlot(i, j)
                    self.grid[(pj, pi)] = PandasTablePlot(self.param_table_data._data, self,  w, pj, pi, model_controls)
                    self.plot_names[w] = (pj, pi)

        for i, pi in enumerate(model.params):
            self.grid[(self.model.params[0], pi)].view.setLabel('left', self.model.short_names[pi])
            self.grid[(pi, self.model.params[-1])].view.setLabel('bottom', self.model.short_names[pi])

        self.hover_plot = []
        self.hover_set = set()

        n = len(self.model.params)
        k = n//2
        self.big_plot_view = self.scatterplot_matrix.addPlot(0, k + n % 2, k, k)
        self.big_plot = None
        self.big_plot_index = None

        self.scatterplot_matrix.sceneObj.sigMouseHover.connect(self.hover)
        self.scatterplot_matrix.sceneObj.sigMouseClicked.connect(self.clicked)

        self.param_table.hoverIndexChanged.connect(self.on_table_hover)

        self.splitter.addWidget(self.param_table)
        self.splitter.addWidget(self.scatterplot_matrix)
        self.layout().addWidget(self.splitter)

        if model_controls:
            model_controls.rangeChanged.connect(self.range_changed_listener)

    def range_changed_listener(self, event):
        for p in self.grid.values():
            p.set_ranges()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:

        if event.key() in [QtCore.Qt.Key_Delete]:
            print('Pressing key:', event.key() )

            selected = self.param_table.selectionModel().selectedRows()
            self.param_table.selectionModel().clearSelection()
            self.param_table_data.delete_selected(selected)

            self.update_plots()

    def show_selection(self, selected_ids):

        for w in self.grid.values():
            w.highlight_id(selected_ids)

        if self.big_plot:
            self.big_plot.highlight_id(selected_ids)

    def on_plot_hover(self, id_list):

        self.notify_plots_to_hover(id_list)
        self.param_points_hovered.emit(id_list)

    def on_plot_click(self, id_list):

        if len(id_list) < 1:
            return

        ids = id_list[0]

        # mapper = SelectionColorMapper(ids, ids)
        # self.selection_color_mapper = mapper
        # print('On plot click:', self.selection_color_mapper.colors.keys())
        # self.show_selection(ids)
        # self.tracer_point_selected.emit(ids[0])

        self.param_table.selectionModel().select(
            self.param_table_data.index(self.param_table_data.get_index_by_id(ids), 0, QModelIndex()),
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        )

    def on_table_hover(self, index):
        """
        Callabck for handling the hover in the table view
        :param index: hovered index in the table
        :return: None
        """
        if index.isValid():
            id_list = [self.param_table_data.get_selected_row(index)]
            # id_list = [index.row()]

            self.notify_plots_to_hover( id_list )
            self.param_points_hovered.emit(id_list)

    def table_selection_changed(self, sel, desel):
        """
        Callback for the table view selection changed.

        We need to update the list of currently selected
        points and notify the rest of the app.

        :param sel: Newly selected rows in the table
        :param desel: Deselected row in the table
        :return: None
        """

        # print('Selection changed')
        selection = self.param_table.selectionModel().selectedRows()
        ids, values = self.param_table_data.get_sorted_selection(selection)

        mapper = SelectionColorMapper(ids, values)
        self.selection_color_mapper = mapper

        self.show_selection(ids)
        self.tracer_points_selected.emit(ids, self.selection_color_mapper)

    def row_in_table_selected(self, index: QModelIndex):
        print(index.row())
        obj_id = self.param_table_data.get_selected_row(index)

        self.show_selection([obj_id])
        self.tracer_point_selected.emit(obj_id)

    def clicked(self, event):

        if event.double():
            if len(self.hover_plot) == 1:

                if self.hover_plot[0] == self.big_plot_view:
                    return
                small_plot_view = self.hover_plot[0]
                (pi, pj) = self.plot_names[self.hover_plot[0]]
                # small_plot = self.grid[(pi, pj)]

                self.big_plot = PandasTablePlot(self.param_table_data._data, self, self.big_plot_view, pi, pj, model_controls=self.model_controls)

                # self.big_plot.duplicate_plot(small_plot)
                self.big_plot.update_plot()

                self.big_plot_view.setLabel('left', self.model.names[pj])
                self.big_plot_view.setLabel('bottom', self.model.names[pi])

                # self.big_plot.view.setXLink(small_plot_view)
                # self.big_plot.view.setYLink(small_plot_view)

        # else:
        #     items = self.scatterplot_matrix.scene().items(event.scenePos())
        #
        #     for e in items:
        #         print('Clicked at:', e)
        #         if isinstance(e, ScatterPlotItem):
        #             ids = [e.data[0][7]]
        #
        #             self.selection_color_mapper = SelectionColorMapper(ids, ids)
        #             self.show_selection(ids)
        #             self.param_table.selectRow(e.data[0][7])
        #
        #             # Emit the selected id
        #             self.tracer_point_selected.emit(e.data[0][7])
        #             break

    def notify_plots_to_hover(self, ids):
        for p in self.grid.values():
            p.hover_id(ids)

        if self.big_plot:
            self.big_plot.hover_id(ids)

    def hover(self, objects):

        self.hover_plot = []

        ids = set()

        for e in objects:
            if isinstance(e, ScatterPlotItem):
                    ids.add(e.data[0][7])

            if isinstance(e, PlotItem):
                self.hover_plot.append(e)

        if ids != self.hover_set:

            self.hover_set = ids
            self.notify_plots_to_hover(ids)
            self.param_points_hovered.emit(ids)

    def update_plots(self):

        if self.big_plot:
            self.big_plot.update_plot()

        for plot in self.grid.values():
            plot.update_plot()

    def paint_param_point(self, tracer_obj: TracerParamObject):

        assert tracer_obj is not None
        assert tracer_obj.id is not None

        self.param_table_data.add_entry(tracer_obj)
        
        t = time.process_time_ns()
        self.param_table.repaint()
        print(f'Param table repainted in {(time.process_time_ns() - t) * 1e-6:.1f} ms')

        t = time.process_time_ns()
        self.update_plots()
        print(f'Param plots repainted in {(time.process_time_ns() - t) * 1e-6:.1f} ms')

        # self.param_table.selectRow(tracer_obj.id)
        # self.selection_color_mapper = SelectionColorMapper([tracer_obj.id], [tracer_obj.id])
        # self.show_selection([tracer_obj.id])


#########################################################################################################
# Param Mapping class
#
class PramMapping(QtWidgets.QWidget):

    def __init__(self):
        super(PramMapping, self).__init__()

        # Init the model and the solver
        self.model = QuadraticModel()
        self.iso_trotter = IsoTrotter(self.model)
        self.tracer_db = TracerObjectDatabase()

        # Main view layout
        self.pyplot_view = MplWidget(dpi=100)
        self.pyplot_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pyplot_view.toolbar.show()
        self.pyplot_view.sc.fig.tight_layout()
        self.pyplot_view.sc.fig.set_size_inches(10.5, 10.5)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.pyplot_view)

        # Hooking the click fit method
        self.click_fitter = ClickFitter(self, self.model, self.pyplot_view, )

        self.controls = RealModelControlsWidget(self.model)
        self.controls.paramsChanged.connect(self.on_params_change)
        layout.addWidget(self.controls)

        self.button = QPushButton('Rotate')
        self.button.clicked.connect(self.rotate_params)
        layout.addWidget(self.button)

        self.button = QPushButton('Reset')
        self.button.clicked.connect(self.reset_rotation)
        layout.addWidget(self.button)

        self.iso_tracer = IsoTracer(model=self.model)

        self.iso_tracer.param_points_hovered.connect(self.show_hovered_points)
        self.iso_tracer.tracer_point_selected.connect(self.reset_to_old_result)
        self.iso_tracer.tracer_points_selected.connect(self.show_selected_points)

        w = QWidget()
        w.setLayout(layout)
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.iso_tracer)
        self.layout().addWidget(w)

        # self.iso_tracer.show()

        # Whatever
        self.model_curves = []
        self.hover_model_curves = []
        self.selected_model_curves = []

        self.m = np.eye(3)
        self.mapping = np.array([
            .0001, 0, 0,
            0, -0.0001, 0,
            0,  0, 20
        ]).reshape((3,3))

        self.params = {'a': 0.01,
                       'b': -0.002,
                       'c': 700}
        # self.set_sliders(self.params)
        self.controls.set_params(self.params)

        self.pyplot_view.sc.axes.clear()
        self.pyplot_view.sc.axes.set_xlim(0, 10)
        self.plot_model(*self.model.solve(self.params))

    def linear_cmap(self, val):
        cmap = plt.get_cmap('Reds')
        return cmap(val)

    def show_hovered_points(self, tracer_points):

        self.show_old_model_run(tracer_points)

    def show_selected_points(self, tracer_points_ids, color_mapper):

        for e in self.selected_model_curves:
            e.remove()
        self.selected_model_curves = []

        for i, pid in enumerate(tracer_points_ids):
            # c = color_mapper.get_color(pid)
            # cc = (c.redF(), c.greenF(), c.blueF())

            p = self.tracer_db.find_by_id(pid)
            z, w = p.result
            self.selected_model_curves += self.pyplot_view.sc.axes.plot(w, z, c=color_mapper.get_mpl_color(pid))

        self.pyplot_view.sc.draw()

    def reset_to_old_result(self, id):

        tracer_obj = self.tracer_db.find_by_id(id)

        self.click_fitter.paint_result(tracer_obj)
        self.controls.set_params(tracer_obj.params)
        self.plot_model(*tracer_obj.result)


    def show_old_model_run(self, paramz):

        # print('Showing old model', paramz)
        for e in self.model_curves:
            e.remove()
        self.model_curves = []

        for i, pid in enumerate(paramz):
            p = self.tracer_db.find_by_id(pid)
            z, w = p.result
            self.model_curves += self.pyplot_view.sc.axes.plot(w, z, c='red')

        self.pyplot_view.sc.draw()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        QApplication.quit()

    # def get_params(self):
    #     return self.params
    #
    # def set_params(self, params):
    #     print('Setting params to', params)
    #
    #     self.params = params
    #     self.controls.set_params(params)
    #     # self.set_sliders(params)
    #     self.plot_model(*self.model.solve(self.params))

    def get_ranges(self):
        return self.model.ranges

    def reset_rotation(self):
        self.m = np.eye(3)
        print(self.m)
        self.update_sliders(None)

    def rotate_params(self):
        self.m = ortho_group.rvs(dim=3)
        print(self.m)
        self.update_sliders(None)

    def simple_update_sliders(self, val):
        params = np.array([s.value() for s in self.sliders])

        mapping = np.array([
            0.1, 0, 0,
            0, 50, 0,
            0, 0, 20
        ])
        mapping = mapping.reshape((3, 3))
        p = mapping.dot(self.m.dot(params - 50))

        self.simple_params(
            *tuple(p)
        )

    def click_fitting_targets(self, targets):

        try:
            self.update_(None, self.controls.get_params(), None, targets)
        except RuntimeError as re:
            print(re)
        except LinAlgError as le:
            print(le)

    def on_params_change(self, b, params, variable):

        try:
            self.update_(
                b, params, variable, self.click_fitter.get_targets()
            )
        except RuntimeError as re:
            print(re)
        except LinAlgError as le:
            print(le)

    def update_(self, b, params, variable, targets):
        print('Updating')

        params = self.iso_trotter.isotrotting(params, targets)

        res = self.model.solve(params)

        t = self.tracer_db.create(
            TracerParamObject(
                params=params,
                result=res,
                targets=self.click_fitter.get_targets(),
                variable=variable
            )
        )

        self.iso_tracer.paint_param_point(t)
        self.plot_model(*t.result)
        self.controls.set_params(t.params)

    # def resizeEvent(self, event):
    #     print('resize event happening', event.size())

    def simple_params(self, w_max, z_c, z_max):

        if w_max == 0 or z_c == 0 or z_max == 0:
            return
        a = 0.5 * w_max ** 2 / z_c
        b = - a * z_c / (z_max)

        self.params = {'a': a, 'b': b, 'c': z_c}

        self.plot_model(*self.model.solve(self.params))

    def plot_model(self, z, w):

        for e in self.model_curves:
            e.remove()
        self.model_curves = []
        self.model_curves += self.pyplot_view.sc.axes.plot(w, z, 'k')

        self.pyplot_view.sc.draw()


if __name__ == "__main__":

    import os
    import sys

    ResourceManager(os.path.dirname(os.path.abspath(__file__+'/..')))
    print(ResourceManager().base_path)
    app = QApplication(sys.argv)

    window = PramMapping()
    window.show()
    sys.exit(app.exec_())

