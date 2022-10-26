
import os
import numpy as np
import matplotlib.pyplot as plt

from OpenGL.raw.GL._types import GLfloat, GLuint, GLbyte

from PyQt5.QtGui import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVersionProfile,
    QOpenGLVertexArrayObject,
    QSurfaceFormat,
    QOpenGLTexture, QMatrix4x4,
)
from glidar_analyst.util.resource_manager import ResourceManager

from glidar_analyst.opengl.base_renderer import BaseRenderer


class TerrainRenderer(BaseRenderer):

    terrain = np.array(plt.get_cmap('terrain')(range(256)))
    VERTEX_NORMALS_FOLDER = './opengl/shaders/surface_vert_normals/'

    def __init__(self, gl):

        super(TerrainRenderer, self).__init__(gl)

        self.vertices, self.x0, self.y0 = None, 0, 0

        self.surface_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.surface_vao = QOpenGLVertexArrayObject()
        self.surface_ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        self.surface_normals = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)

        self.surface_len = None
        self.surface_program = None

        self.colorTexture = QOpenGLTexture(QOpenGLTexture.Target1D)
        self.initTexture(self.colorTexture, TerrainRenderer.terrain)

        # self.createSurfaceShaders()
        self.createShadersFromFolder(self.VERTEX_NORMALS_FOLDER)
        self.create_surface_VBO()

    def closeEvent(self):

        print("Destructor calls in closeEvent...")
        self.surface_vao.destroy()
        self.surface_vbo.destroy()
        self.surface_ibo.destroy()
        self.surface_normals.destroy()

        self.colorTexture.destroy()

    def set_surface_data(self, data, normals=None):

        if normals is None:
            self.createSurfaceShaders()

        else:
            self.createShadersFromFolder(self.VERTEX_NORMALS_FOLDER)

        print('Loading to GPU')
        self.vertices, self.x0, self.y0 = data[0], data[1], data[2]

        vertices = self.vertices
        vertex_data = (GLfloat * len(vertices))(*vertices)

        self.surface_vao.bind()

        self.surface_vbo.bind()
        self.surface_vbo.allocate(vertex_data, len(vertices) * 4)
        self.surface_vbo.release()

        if normals is not None:
            normal_data = (GLfloat * len(normals))(*normals)

            self.surface_normals.bind()
            self.gl.glEnableVertexAttribArray(1)
            self.surface_normals.allocate(normal_data, len(normals) * 4)
            self.surface_normals.release()
        # else:
            # self.gl.glDisableVertexAttribArray(1)

        self.surface_vao.release()

        self.surface_len = len(vertex_data)
        print("Data loaded.")

    def create_surface_VBO(self):

        self.surface_vao.create()
        if self.surface_vao.isCreated():
            self.surface_vao.bind()
        else:
            raise RuntimeError("Unable to create VAO.")

        #
        # Vertices
        self.surface_vbo.create()
        self.surface_vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)

        vertices = (100., 0., 200.,
                    0., 100., 200.,
                    100., 0., 200.,
                    0., 100., 100.)

        vertex_data = (GLfloat * len(vertices))(*vertices)
        self.surface_vbo.bind()
        self.surface_vbo.allocate(vertex_data, len(vertices) * 4)

        self.surface_len = len(vertex_data)

        self.gl.glEnableVertexAttribArray(0)
        self.gl.glVertexAttribPointer(0,                    # index (default 0 for position)
                                      3,                    # size of single vertex (3 or 4)
                                      self.gl.GL_FLOAT,     # type
                                      self.gl.GL_FALSE,     # normalized
                                      0,                    # stride
                                      None)                 # pointer   Specifies a offset of the first component.
                                                            #           The initial value is None.
        self.surface_vbo.release()

        #
        # Nomals
        self.surface_normals.create()
        self.surface_normals.setUsagePattern(QOpenGLBuffer.StaticDraw)

        normals  = (0., 0., 1.,
                    0., 0., 1.,
                    0., 0., 1.,
                    0., 0., 1.)

        normal_data = (GLfloat * len(normals))(*normals)
        self.surface_normals.bind()
        self.surface_normals.allocate(normal_data, len(normals) * 4)    # sizeof float

        self.gl.glEnableVertexAttribArray(1)
        self.gl.glVertexAttribPointer(1,  # index (location 1 for normals)
                                      3,  # size of single vertex (3 or 4)
                                      self.gl.GL_FLOAT,  # type
                                      self.gl.GL_FALSE,  # normalized
                                      0,  # stride
                                      None)  # pointer   Specifies a offset of the first component.
                                             #           The initial value is None.
        self.surface_normals.release()

        self.surface_vao.release()

    def createSurfaceShaders(self):

        self.surface_program = QOpenGLShaderProgram()       # Possibly put parent here...
        folder = ResourceManager().get_absolute_path('')

        ###########################################################
        # VERTEX SHADER
        if self.surface_program.addShaderFromSourceCode(
                QOpenGLShader.Vertex, self.read_file(folder + './opengl/shaders/surface_geom_normals/vert.glsl')):
            print('Vertex shader compiled succesfully.')
        else:
            print('Error compiling vertex shader.')
            print(self.surface_program.log())
            raise RuntimeError('Vertex shader.')

        ###########################################################
        # GEOMETRY SHADER
        if self.surface_program.addShaderFromSourceCode(
                QOpenGLShader.Geometry, self.read_file(folder + './opengl/shaders/surface_geom_normals/geom.glsl')):
            print('Vertex shader compiled succesfully.')
        else:
            print('Error compiling geometry shader.')
            print(self.surface_program.log())
            raise RuntimeError('Geometry shader.')

        ###########################################################
        # FRAGMENT SHADER
        if self.surface_program.addShaderFromSourceCode(
                QOpenGLShader.Fragment, self.read_file(folder + './opengl/shaders/surface_geom_normals/frag.glsl')):
            print('Fragment shader compiled successfully.')
        else:
            print('Error compiling fragment shader.')
            print(self.surface_program.log())
            raise RuntimeError('Fragment shader.')

        ###########################################################
        # LINKING SHADERS
        if self.surface_program.link():
            print('Program linked successfully.')
        else:
            print('Linking error.')
            self.surface_program.log()
            raise RuntimeError('Linking shaders.')

    def draw_surface(self, mvp):

        if self.surface_len is None:
            return

        self.surface_program.bind()

        self.gl.glActiveTexture(self.gl.GL_TEXTURE3)
        self.colorTexture.bind()

        self.surface_program.setUniformValue(self.surface_program.uniformLocation('terrainSampler'), 3)
        self.surface_program.setUniformValue(self.surface_program.uniformLocation('modelViewProjection'), mvp)

        self.surface_vao.bind()
        if self.surface_vbo.bind():
            self.gl.glDrawArrays(self.gl.GL_TRIANGLE_STRIP, 0, self.surface_len)

        self.surface_vbo.release()
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
