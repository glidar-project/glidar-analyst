
import os
import pyproj
import numpy as np
import matplotlib.pyplot as plt

from OpenGL.raw.GL._types import GLfloat, GLuint, GLbyte, GLint
from PyQt5.QtCore import Qt

from PyQt5.QtGui import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVersionProfile,
    QOpenGLVertexArrayObject,
    QSurfaceFormat,
    QOpenGLTexture, QMatrix4x4, QImage, QKeyEvent,
)

from glidar_analyst.opengl.base_renderer import BaseRenderer
from glidar_analyst.opengl.terrain_renderer import TerrainRenderer


class EarthRenderer(BaseRenderer):

    SHADERS_FOLDER = './opengl/shaders/earth/'
    ECEF = pyproj.Proj(proj='geocent')
    LLH = pyproj.Proj(proj='latlong')

    RADIUS = 6378.

    def __init__(self, gl):

        super(EarthRenderer, self).__init__(gl)

        self.surface_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.surface_vao = QOpenGLVertexArrayObject()
        self.surface_ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        self.surface_normals = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)

        self.surface_len = None
        self.surface_program = None

        self.colorTexture = QOpenGLTexture(QOpenGLTexture.Target1D)
        self.initTexture(self.colorTexture, TerrainRenderer.terrain)

        self.earth_texture = QOpenGLTexture(QImage("./opengl/textures/8081_earthmap10k.jpg"))

        self.createShadersFromFolder(self.SHADERS_FOLDER)
        self.create_surface_VBO()

    def closeEvent(self):

        print("Destructor calls in closeEvent...")
        self.surface_vao.destroy()
        self.surface_vbo.destroy()
        self.surface_ibo.destroy()
        self.surface_normals.destroy()

        self.colorTexture.destroy()

    def LLH_to_ECEF(self, lat, lon, alt):
        x, y, z = pyproj.transform(self.LLH, self.ECEF, lon, lat, alt, radians=False)
        return x/1000000, y/1000000, z/1000000

    def create_surface_VBO(self):

        self.surface_vao.create()
        if self.surface_vao.isCreated():
            self.surface_vao.bind()
        else:
            raise RuntimeError("Unable to create VAO.")

        lat = np.linspace(-np.pi, np.pi, 30)
        lon = np.linspace(0, 2 * np.pi, 60)
        grid_lat, grid_lon = np.meshgrid(lat, lon)

        grid_x = np.cos(grid_lat) * np.cos(grid_lon)
        grid_y = np.cos(grid_lat) * np.sin(grid_lon)
        grid_z = np.sin(grid_lat)

        normals = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1)
        vertices = self.RADIUS * normals.copy()

        indices = np.arange(grid_x.size)
        indices.shape = grid_x.shape

        triangle_indices = []
        for j in range(indices.shape[1] - 1):
            for i in range(indices.shape[0] - 1):
                triangle_indices.append(indices[i, j])
                triangle_indices.append(indices[i, j + 1])
                triangle_indices.append(indices[i + 1, j])

                triangle_indices.append(indices[i + 1, j + 1])
                triangle_indices.append(indices[i + 1, j])
                triangle_indices.append(indices[i, j + 1])

        triangle_indices = np.array(triangle_indices).flatten().tolist()
        print(triangle_indices)

        vertices = vertices.flatten().tolist()

        #
        # Vertices
        self.surface_vbo.create()
        self.surface_vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)

        vertex_data = (GLfloat * len(vertices))(*vertices)

        self.surface_vbo.bind()
        self.surface_vbo.allocate(vertex_data, len(vertices) * 4)   # sizeof float

        self.surface_len = len(vertex_data)

        self.gl.glEnableVertexAttribArray(0)
        self.gl.glVertexAttribPointer(0,                    # index (default 0 for position)
                                      3,                    # size of single vertex (3 or 4)
                                      self.gl.GL_FLOAT,     # type
                                      self.gl.GL_FALSE,     # normalized
                                      0,                    # stride
                                      None)                 # pointer   Specifies a offset of the first component.
                                                            #           The initial value is None.

        #
        # Nomals
        self.surface_normals.create()
        self.surface_normals.setUsagePattern(QOpenGLBuffer.StaticDraw)

        normals = normals.flatten().tolist()
        normal_data = (GLfloat * len(normals))(*normals)

        self.surface_normals.bind()
        self.surface_normals.allocate(normal_data, len(normals) * 4)    # sizeof float

        self.gl.glEnableVertexAttribArray(1)
        self.gl.glVertexAttribPointer(1,  # index (location 1 for normals)
                                      3,  # size of single vertex (3 or 4)
                                      self.gl.GL_FLOAT,  # type
                                      self.gl.GL_TRUE,  # normalized
                                      0,  # stride
                                      None)  # pointer   Specifies a offset of the first component.
                                             #           The initial value is None.

        #
        # Index
        self.surface_ibo.create()
        self.surface_ibo.setUsagePattern(QOpenGLBuffer.StaticDraw)

        index_data = (GLuint * len(triangle_indices))(*triangle_indices)

        self.surface_ibo.bind()
        self.surface_ibo.allocate(index_data, len(triangle_indices) * 4)    # sizeof int

        self.surface_ibo.release()
        self.surface_vbo.release()
        self.surface_normals.release()

        self.surface_vao.release()

    def draw_surface(self, mvp):

        if self.surface_len is None:
            return

        self.surface_program.bind()

        self.gl.glActiveTexture(self.gl.GL_TEXTURE3)
        self.colorTexture.bind()

        self.gl.glActiveTexture(self.gl.GL_TEXTURE0)
        self.earth_texture.bind()

        self.surface_program.setUniformValue(self.surface_program.uniformLocation('textureSampler'), 0)
        self.surface_program.setUniformValue(self.surface_program.uniformLocation('terrainSampler'), 3)
        self.surface_program.setUniformValue(self.surface_program.uniformLocation('modelViewProjection'), mvp)

        self.surface_vao.bind()
        if self.surface_vbo.bind() and self.surface_normals.bind() and self.surface_ibo.bind():
            # self.gl.glDrawArrays(self.gl.GL_TRIANGLE_STRIP, 0, self.surface_len)

            self.gl.glDrawElements(
                self.gl.GL_TRIANGLES,  # mode
                self.surface_ibo.size(),  # count
                self.gl.GL_UNSIGNED_INT,  # type
                None  # element array buffer offset
            )

        self.surface_vbo.release()
        self.surface_normals.release()
        self.surface_ibo.release()
        self.surface_vao.release()
        self.colorTexture.release()
        self.surface_program.release()

    def display(self, camera):

        ###########################################################
        # Linear algebra
        modelMatrix = QMatrix4x4()
        viewMatrix = camera.view_matrix()
        mvp = camera.projectionMatrix * viewMatrix * modelMatrix

        self.draw_surface(mvp)

    def keyPressedEvent(self, event):

        print(event)
        if event.key() == Qt.Key_F5:
            self.createShadersFromFolder(self.SHADERS_FOLDER)


