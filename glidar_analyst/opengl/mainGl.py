import numpy

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QMainWindow, QApplication, QOpenGLWidget, QSlider
from PyQt5.QtGui import QMatrix4x4
from numpy import random

import OpenGL.GL as gl

from glidar_analyst.gui.my_base_widget import MyBaseWidget
from glidar_analyst.opengl.camera import Camera
from glidar_analyst.opengl.earth_renderer import EarthRenderer


class ControlsWidget(MyBaseWidget):

    def __init__(self, main_win):
        super(ControlsWidget, self).__init__(main_win)

        self.main_gl_window = main_win
        self.slider = QSlider(Qt.Horizontal, self)
        self.params = { 'threshold': 0.5 }
        self.slider.valueChanged.connect(self.slider_changed)
        self.slider.setTracking(True)

    def slider_changed(self, val):
        self.params['threshold'] = 0.5 + val * 0.005
        self.main_gl_window.update_vis_params(self.params)


class OpenGLWidget(QOpenGLWidget):

    def __init__(self, *args, **kwargs):
        """Initialize OpenGL version profile."""
        super(OpenGLWidget, self).__init__(*args, **kwargs)

        self.setFocusPolicy(Qt.ClickFocus)

        self.model_control = ControlsWidget(self)
        self.model_control.setWindowFlags(Qt.WindowStaysOnTopHint)      # Necessary?
        self.model_control.setVisible(True)

        self.gl = None

        self.camera = Camera(self)
        self.renderers = []
        self.gl_init_listeners = []

    def close(self) -> bool:
        self.model_control.close()
        super(OpenGLWidget, self).close()

    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)
        print("Destructor calls in closeEvent...")

        for r in self.renderers:
            r.closeEvent()

    def minimumSizeHint(self):
        return QSize(300, 150)

    def mousePressEvent(self, event):
        self.camera.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.camera.mouseMoveEvent(event)
        self.repaint()

    def keyPressEvent(self, QKeyEvent):
        print("Key pressed", QKeyEvent.key())
        for r in self.renderers:
            r.keyPressedEvent(QKeyEvent)

    #
    # OpenGL part
    #
    def getOpenglInfo(self):
        info = """
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
        """.format(
            gl.glGetString(gl.GL_VENDOR),
            gl.glGetString(gl.GL_RENDERER),
            gl.glGetString(gl.GL_VERSION),
            gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
        )

        return info

    def initializeGL(self):
        """Apply OpenGL version profile and initialize OpenGL functions."""
        self.gl = gl

        print('Initializing opnegl...')
        print(self.getOpenglInfo())

        self.gl.glEnable(self.gl.GL_DEPTH_TEST)
        self.gl.glDisable(self.gl.GL_CULL_FACE)
        self.gl.glClearColor(0.0, 0.0, 0.0, 0.0)

        for e in self.gl_init_listeners:
            e.init_gl()

    def paintGL(self):

        if not self.gl:
            return
        # print(gl.glGetError())

        self.gl.glClearColor(228./255., 238./255., 241./255., 1.)
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)

        for r in self.renderers:
            r.display(self.camera)

    def resizeGL(self, w, h):
        """Resize viewport to match widget dimensions."""
        # self.gl.glViewport(0, 0, w, h)
        retinaScale = self.devicePixelRatio()
        self.gl.glViewport(0, 0, self.width() * retinaScale, self.height() * retinaScale)

        projectionMatrix = QMatrix4x4()
        projectionMatrix.perspective(60.0, self.width() / self.height(), 0.01, 100.0)

        self.camera.projectionMatrix = projectionMatrix

    #
    # Custom stuff
    #
    def add_renderer(self, renderer):
        self.renderers.append(renderer)

    def update_vis_params(self, params):
        self.repaint()


class QTWithGLTest(QMainWindow):
    """Main window."""

    def __init__(self, *args, **kwargs):
        """Initialize with an OpenGL Widget."""
        super(QTWithGLTest, self).__init__(*args, **kwargs)

        self.widget = OpenGLWidget()
        # self.widget.setFocusPolicy(Qt.ClickFocus)
        self.setCentralWidget(self.widget)
        self.show()


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = QTWithGLTest()
    window.show()

    positions = numpy.stack((list(
                              [random.uniform(-100, 100),
                              random.uniform(-100, 100),
                              random.uniform(-100, 100),
                              i/100] for i in range(100))))

    er = EarthRenderer(window.widget.gl)
    window.widget.add_renderer(er)

    # th = ThermalRenderer(window.widget)
    # window.widget.add_renderer(th)
    #
    # tr = TerrainRenderer(window.widget.gl)
    # window.widget.add_renderer(tr)
    #
    # th.update_thermal_data(positions)
    #
    # file_abspath = r'C:\\Users\\Juraj\\Work\\convection-analysis\\data\\Charts\\Norgeskart\\DEM\\6703_2_10m_z32.dem'
    # tr.set_surface_data(*load_dem_surface(file_abspath, 20))

    sys.exit(app.exec_())