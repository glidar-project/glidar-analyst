import random

import numpy as np
import pandas as pd

import matplotlib
from PyQt5.QtWidgets import QVBoxLayout, QApplication
from glidar_analyst.util.resource_manager import ResourceManager

from glidar_analyst.opengl.dem_loader import convert_file, make_triangle_strip
from glidar_analyst.opengl.terrain_renderer import TerrainRenderer
from glidar_analyst.opengl.thermal_renderer import ThermalRenderer
from glidar_analyst.worker import Worker
from glidar_analyst.opengl.mainGl import OpenGLWidget
from glidar_analyst.gui.my_base_widget import MyBaseWidget

matplotlib.use('Qt5Agg')


class Thermal3DView(MyBaseWidget):

    def __init__(self, main_window, *args, **kwargs):

        super(Thermal3DView, self).__init__(*args, **kwargs)

        self.main_window = main_window

        self.w = OpenGLWidget()
        self.w.gl_init_listeners += [self]

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.w)
        self.thermal = None
        self.surface = None
        self.thermal_renderer = None
        self.terrain_renderer = None

    def init_gl(self):

        # Thermal Init
        self.thermal_renderer = ThermalRenderer(self.w)
        self.w.add_renderer(self.thermal_renderer)

        # Terrain Init
        self.terrain_renderer = TerrainRenderer(self.w.gl)
        self.w.add_renderer(self.terrain_renderer)

        self.load_surface()

    def get_altitude(self, x, y):

        if not self.surface:
            return 0

        grid, x0, y0, x1, y1 = self.surface

        if x > x1 or x < x0 or y > y1 or y < y0:
            return 0

        dx = x1 - x0
        dy = y1 - y0

        i = int((x - x0) / dx * (grid.shape[0] - 1))
        j = int((y - y0) / dy * (grid.shape[1] - 1))

        return grid[i, j] / 10.0

    def load_surface(self):

        if self.main_window is None:
            return

        def result_fn(result):
            surface, strip = result

            self.terrain_renderer.set_surface_data(strip)
            self.surface = surface

        def worker_fn():
            from glidar_analyst.opengl.dem_loader import make_triangle_strip, convert_file
            file_abspath = r'../data/Charts/Norgeskart/DEM/6703_2_10m_z32.dem'
            surface = convert_file(ResourceManager().get_absolute_path(file_abspath))
            strip = make_triangle_strip(*surface, 5)
            return surface, strip

        w = Worker(worker_fn)
        w.signals.result.connect(result_fn)
        self.main_window.threadpool.start(w)

    def select_points(self, selection):

        if self.thermal_renderer is None:
            return

        # TODO: Seems like the indices are off, check for errors
        print(selection)
        print(self.thermal.index)
        self.thermal_renderer.set_selected(selection)
        self.repaint()

    def showThermal(self, df: pd.DataFrame) -> None:

        if self.thermal_renderer is None:
            return

        if not hasattr(df, 'x'):
            return

        X = df.x.mean()
        Y = df.y.mean()

        self.thermal = df
        # if False:
        if X - self.terrain_renderer.x0 < 0 or Y - self.terrain_renderer.y0 < 0:
            data = np.array([
                df['x'].to_numpy() - df.x.mean() + self.terrain_renderer.x0,
                df['y'].to_numpy() - df.y.mean() + self.terrain_renderer.y0,
                df['altitude'].to_numpy(),
                0.5 + 0.2 * df['vario'].to_numpy()
            ]).T
        else:
            data = np.array([
                df['x'].to_numpy() - self.terrain_renderer.x0,
                df['y'].to_numpy() - self.terrain_renderer.y0,
                df['altitude'].to_numpy(),
                0.5 + 0.2 * df['vario'].to_numpy()
            ]).T

        self.thermal_renderer.update_thermal_data(data)

        self.w.camera.lookAt(X - self.terrain_renderer.x0,
                             Y - self.terrain_renderer.y0)
        self.repaint()


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = Thermal3DView(None)
    window.show()

    positions = np.stack(([random.uniform(-100, 100),
                              random.uniform(-100, 100),
                              random.uniform(-100, 100),
                              i / 100] for i in range(100)))


    tr = TerrainRenderer(window.w.gl)
    file_abspath = r'/data/Charts/Norgeskart/DEM/6703_2_10m_z32.dem'
    tr.set_surface_data(make_triangle_strip(*convert_file(file_abspath), 5))
    window.w.add_renderer(tr)

    sys.exit(app.exec_())