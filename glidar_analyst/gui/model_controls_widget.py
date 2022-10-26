import matplotlib
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QPainter
from PyQt5.QtWidgets import QLabel, QVBoxLayout, \
    QSlider, QGridLayout, QLineEdit, QStyle, QInputDialog, QWidget, QHBoxLayout, QSpacerItem, QSizePolicy, QScrollArea
from metpy.units import DimensionalityError, UndefinedUnitError

from glidar_analyst.gui.my_base_widget import MyBaseWidget

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui

import numpy as np


class CustomSlider(QSlider):

    def __init__(self, val, *args, **kwargs):
        super(CustomSlider, self).__init__(*args, **kwargs)

        self.default_value_ = val

    def paintEvent(self, QPaintEvent):

        position = QStyle.sliderPositionFromValue(self.minimum(),
                                                  self.maximum(),
                                                  self.default_value_,
                                                  self.width())
        # print(self.minimum(), self.maximum(), position, self.width())

        painter = QPainter(self)
        painter.drawLine(position, 0, position, self.height())
        super(CustomSlider, self).paintEvent(QPaintEvent)


class ClickableLabel(QLabel):

    doubleclicked = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(ClickableLabel, self).__init__(*args, **kwargs)

    def mouseDoubleClickEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.doubleclicked.emit()


class SliderHelper(MyBaseWidget):

    enterSignal = pyqtSignal(object)
    leaveSignal = pyqtSignal(object)
    rangeChanged = pyqtSignal(object)

    def in_base(self, val):
        if self.current_units is None:
            return val
        return (val * self.current_units).to(self.base_units).magnitude

    def in_current(self, val):
        if self.current_units is None:
            return val
        return (val * self.base_units).to(self.current_units).magnitude

    def get_range(self):
        return self.in_base(float(self.minEdit.text())), self.in_base(float(self.maxEdit.text()))

    @property
    def range(self):
        return self.get_range()

    @property
    def value(self):
        return self.value_

    @value.setter
    def value(self, val):
        self.value_ = val

    def __init__(self, id, range, value, callback, name, units=None, n_steps=1001, *args, **kwargs):
        
        super(SliderHelper, self).__init__(*args, **kwargs)

        self.id = id
        self.value_ = value
        self.callback = callback
        self.name = name
        self.selected = None
        self.n_steps = n_steps
        self.default_value = value
        self.intervalFraction = 0.05
        self.base_units = units
        self.current_units = units

        self.minEdit = QLineEdit(repr(range[0]), self)
        self.maxEdit = QLineEdit(repr(range[1]), self)
        interval = self.intervalFraction * (range[1] - range[0])

        layout = QGridLayout(self)
        # layout.setContentsMargins(0,0,0,0)
        self.nameLabel = None
        if self.current_units is None:
            self.nameLabel = ClickableLabel(name)
        else:
            self.nameLabel = ClickableLabel(name + ' [{}]'.format(self.current_units))
            self.nameLabel.doubleclicked.connect(self.change_units_action)

        self.valueLabel = QLabel(repr(self.value))
        self.valueLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.intervalLabel = QLabel('+- {:0.2f}'.format(interval))

        label_widget = QWidget()
        label_widget.setLayout(QHBoxLayout())

        label_widget.layout().addWidget(self.nameLabel)
        label_widget.layout().addItem(QSpacerItem(0,0, QSizePolicy.Expanding, QSizePolicy.Expanding))
        label_widget.layout().addWidget(self.valueLabel)
        label_widget.layout().addWidget(self.intervalLabel)
        label_widget.layout().setContentsMargins(0,0,0,0)
        layout.addWidget(label_widget)

        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setTracking(True)
        self.slider.setRange(0, n_steps)
        self.slider.setValue(self.val_to_int(self.value))
        self.slider.valueChanged.connect(self.slider_change)
        self.slider.sliderReleased.connect(self.slider_set)
        self.slider.setTickPosition(QSlider.NoTicks)
        layout.addWidget(self.slider, 1, 0, 1, 3)

        edit_widget = QWidget()
        edit_widget.setLayout(QHBoxLayout())
        edit_widget.layout().setContentsMargins(0,0,0,0)
        edit_widget.layout().addWidget(self.minEdit)
        edit_widget.layout().addWidget(self.maxEdit)

        layout.addWidget(edit_widget, 2,0,1,3)
        # layout.addWidget(self.minEdit)
        # layout.addWidget(self.maxEdit)
        self.minEdit.setValidator(QDoubleValidator(-10000, 10000, 5, self))
        self.maxEdit.setValidator(QDoubleValidator(-10000, 10000, 5, self))

        self.minEdit.returnPressed.connect(self.rangeEdited)
        self.maxEdit.returnPressed.connect(self.rangeEdited)

    def change_units_action(self):
        u, ok = QInputDialog.getText(self, 'Set Units', 'Units:', text=str(self.base_units))
        print(u)
        if ok:
            self.change_units_to(u)

    def set_default_value(self, val):
        self.default_value = val
        self.slider.default_value_ = self.val_to_int(val)

    def change_units_to(self, new_units):
        
        if self.base_units is None:
            return

        try:
            new_value = (self.value * self.current_units).to(new_units)

            a, b = self.get_range()                         # in base units
            new_min = (a * self.base_units).to(new_units)   # convert to new units
            new_max = (b * self.base_units).to(new_units)

            self.current_units = new_value.units            # only using the units from new_value
            self.minEdit.setText(repr(new_min.magnitude))
            self.maxEdit.setText(repr(new_max.magnitude))

            self.slider_change(self.val_to_int(self.value_))    # update the value label
            self.nameLabel.setText(self.name + ' [{}]'.format(str(self.current_units)))

            self.rangeEdited()
        except (DimensionalityError, UndefinedUnitError) as de:
            print('Units are not compatible...', de)
            return

    def rangeEdited(self):
        min, max = self.get_range()

        interval = self.intervalFraction * (self.in_current(max) - self.in_current(min)) 
        self.intervalLabel.setText('+- {:0.2f}'.format(interval))

        self.slider.setValue(self.val_to_int(self.value))
        self.rangeChanged.emit({self.id: (min, max)})

    def enterEvent(self, event):
        self.enterSignal.emit(self.id)
        self.setStyleSheet("background-color:#F8F8F8;")

    def leaveEvent(self, event):
        self.leaveSignal.emit(self.id)
        self.setStyleSheet("")

    def set_value(self, val):
        self.slider.setValue(self.val_to_int(val))

    def int_to_val(self, n):
        a, b = self.get_range()
        return a + (b-a) * (n / (self.n_steps - 1))

    def val_to_int(self, val):
        a, b = self.get_range()
        if b-a == 0:
            return 0
        return int(np.round((self.n_steps - 1) * (val - a) / (b - a)))

    def slider_change(self, value):
        self.value = self.int_to_val(value)
        self.valueLabel.setText('{:.1f}'.format( self.in_current(self.value) ))

        if self.callback is not None:
            self.callback(False, self.id)

    def slider_set(self):
        if self.callback is not None:
            self.callback(True, self.id)


class RealModelControlsWidget(MyBaseWidget):

    paramsChanged = pyqtSignal(bool, object, str)
    rangeChanged = pyqtSignal(object)

    def set_defaults(self, defaults):
        for k in defaults.keys():
            self.sliders[k].set_default_value(defaults[k])

    def __init__(self, model):
        super(RealModelControlsWidget, self).__init__()

        self.model = model

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.sliders = {}
        for k in self.model.params:
            if hasattr(self.model, 'names'):
                name = self.model.names[k]
            else:
                name = k

            units = None
            if hasattr(self.model, 'units'):
                units = self.model.units[k]

            default = 0
            if hasattr(self.model, 'defaults'):
                default = self.model.defaults[k]

            self.sliders[k] = SliderHelper(k, self.model.ranges[k], default, self.callback, name, units=units)

        for s in self.sliders.values():
            layout.addWidget(s)
            s.rangeChanged.connect(self.range_changed_listener)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_area.setMinimumWidth(360)

        content_widget = QWidget()
        content_widget.setLayout(layout)
        content_widget.setContentsMargins(0,0,0,0)
        # content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # content_widget.setStyleSheet('background-color: #003355')
        scroll_area.setWidget(content_widget)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(scroll_area)

    def range_changed_listener(self, event):

        self.rangeChanged.emit(event)

    def get_ranges(self):
        result = {
            k: v.get_range() for k, v in self.sliders.items()
        }
        return result

    def get_params(self):
        result = {
            k: v.value for k, v in self.sliders.items()
        }
        return result

    def set_params(self, val):
        block = self.blockSignals(True)
        for k, s in self.sliders.items():
            s.set_value(val[k])
        self.blockSignals(block)

    def callback(self, final, slider_id):
        self.paramsChanged.emit(final, self.get_params(), slider_id)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(300, 300)


class ModelControlsWidget(RealModelControlsWidget):

    def __init__(self, model):
        
        # model = DummyModel()
        
        super(ModelControlsWidget, self).__init__(model)

        self.sliders['T'].change_units_to('celsius')
        self.sliders['Td'].change_units_to('celsius')

        self.sliders['B'].change_units_to('1 / km')
        self.sliders['C'].change_units_to('1 / km')
        self.sliders['q'].change_units_to('1 / km')

        self.sliders['a'].change_units_to('1 / minute')

