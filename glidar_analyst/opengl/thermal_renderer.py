import numpy
import matplotlib.pyplot as plt

from OpenGL.raw.GL._types import GLfloat, GLuint, GLbyte

from PyQt5.QtGui import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVersionProfile,
    QOpenGLVertexArrayObject,
    QSurfaceFormat,
    QVector2D,
    QVector3D,
    QVector4D,
    QOpenGLTexture, QMatrix4x4,
)
from glidar_analyst.util.resource_manager import ResourceManager

from glidar_analyst.opengl.base_renderer import BaseRenderer


class ThermalRenderer(BaseRenderer):

    divergent_palette = numpy.array(plt.get_cmap('coolwarm')(numpy.arange(0,255)))

    def __init__(self, parent):

        self.parent = parent

        super(ThermalRenderer, self).__init__(self.parent.gl)

        # self.parent.makeCurrent()       # ? maybe helps?

        self.m_vao = QOpenGLVertexArrayObject(self.parent)  # QOpenGLVertexArrayObject
        self.m_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.m_ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        self.selected_ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)

        self.m_program = None  # QOpenGLShaderProgram()
        # self.m_shader = None  # QOpenGLShader()
        # self.m_texture_id = None  # QOpenGLTexture()

        # self.m_texture = QOpenGLTexture(QOpenGLTexture.Target1D)
        self.colormapTexture = QOpenGLTexture(QOpenGLTexture.Target1D)

        # self.m_uniformLocations = {}

        # Uniforms
        self.radiusScale = 10.0
        self.threshold = 0.0
        self.selected = False

        self.buff_len = 0

        self.createShaders()
        self.createVBO()

        self.initTexture(self.colormapTexture, ThermalRenderer.divergent_palette)

    def closeEvent(self):

        self.selected_ibo.destroy()

        self.m_vbo.destroy()
        self.m_ibo.destroy()
        self.m_vao.destroy()

        # self.m_texture.destroy()
        self.colormapTexture.destroy()

    def createShaders(self):

        self.m_program = QOpenGLShaderProgram(self.parent)

        folder = ResourceManager().get_absolute_path('')

        ###########################################################
        # VERTEX SHADER
        if self.m_program.addShaderFromSourceCode(
                QOpenGLShader.Vertex, self.read_file(folder + '/opengl/shaders/sphere/sphere-vs.glsl')):
            print('Vertex shader compiled succesfully.')
        else:
            print('Error compiling vertex shader.')
            print(self.m_program.log())
            raise RuntimeError('Vertex shader.')

        ###########################################################
        # GEOMETRY SHADER
        if self.m_program.addShaderFromSourceCode(
                QOpenGLShader.Geometry, self.read_file(folder + '/opengl/shaders/sphere/sphere-gs.glsl')):
            print('Vertex shader compiled succesfully.')
        else:
            print('Error compiling geometry shader.')
            print(self.m_program.log())
            raise RuntimeError('Geometry shader.')

        ###########################################################
        # FRAGMENT SHADER
        if self.m_program.addShaderFromSourceCode(
                QOpenGLShader.Fragment, self.read_file(folder + '/opengl/shaders/sphere/sphere-fs.glsl')):
            print('Fragment shader compiled successfully.')
        else:
            print('Error compiling fragment shader.')
            print(self.m_program.log())
            raise RuntimeError('Fragment shader.')

        ###########################################################
        # LINKING SHADERS
        if self.m_program.link():
            print('Program linked successfully.')
        else:
            print('Linking error.')
            self.m_program.log()
            raise RuntimeError('Linking shaders.')

        # for name in ['modelViewMatrix', 'projectionMatrix', 'radiusScale',
        #              'modelViewProjectionMatrix', 'inverseModelViewProjectionMatrix',
        #              'normalMatrix', 'atomDataSampler']:
        #     self.m_uniformLocations[name] = self.m_program.uniformLocation(name)

    def createVBO(self):

        vertices = [10, 10, 10, 0.5,
                    10.0, 10., 0, .7,
                    -10.0, 0, 0., .6,
                    10.0, -10, -10., .5]

        vertex_data = (GLfloat * len(vertices))(*vertices)

        indices = (GLuint * 4)(*list(range(4)))

        self.m_vao.create()
        if self.m_vao.isCreated():
            self.m_vao.bind()
        else:
            raise RuntimeError("Unable to create VAO.")

        self.m_vbo.create()
        self.m_vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        self.m_vbo.bind()
        self.m_vbo.allocate(vertex_data, len(vertices) * 4)

        self.m_ibo.create()
        self.m_ibo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        self.m_ibo.bind()
        self.m_ibo.allocate(indices, len(indices) * 4)

        self.selected = True
        self.selected_ibo.create()
        self.selected_ibo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        self.selected_ibo.bind()
        self.selected_ibo.allocate(indices, len(indices) * 4)

        self.gl.glEnableVertexAttribArray(0)
        self.gl.glVertexAttribPointer(0,                    # index (default 0 for position)
                                      4,                    # size of single vertex (3 or 4)
                                      self.gl.GL_FLOAT,     # type
                                      self.gl.GL_FALSE,     # normalized
                                      0,                    # stride
                                      None)                 # pointer   Specifies a offset of the first component.
                                                            #           The initial value is None.

        self.m_ibo.release()
        self.m_vbo.release()
        self.m_vao.release()

        print(self.gl.glGetError())

        self.buff_len = 4

    def draw_thermal(self, mv, mvp, imvp, normalMatrix, projectionMatrix):

        # self.parent.makeCurrent()

        if not self.m_program.bind():
            print('Could not bind program.')
            return

        # for k, v in self.model_control.params.items():
        #     self.m_program.setUniformValue(self.m_program.uniformLocation(k), v)

        self.m_program.setUniformValue(self.m_program.uniformLocation('projectionMatrix'), projectionMatrix)
        self.m_program.setUniformValue(self.m_program.uniformLocation('modelViewMatrix'), mv)
        self.m_program.setUniformValue(self.m_program.uniformLocation('modelViewProjectionMatrix'), mvp)
        self.m_program.setUniformValue(self.m_program.uniformLocation('inverseModelViewProjectionMatrix'), imvp)
        self.m_program.setUniformValue(self.m_program.uniformLocation('normalMatrix'), normalMatrix)

        self.m_program.setUniformValue(self.m_program.uniformLocation('threshold'), self.threshold)
        self.m_program.setUniformValue(self.m_program.uniformLocation('radiusScale'), self.radiusScale)
        self.m_program.setUniformValue(self.m_program.uniformLocation('selected'), False)

        # print(self.gl.glGetError())
        # self.gl.glActiveTexture(self.gl.GL_TEXTURE0)
        # self.m_texture.bind()
        # self.m_program.setUniformValue(self.m_program.uniformLocation('atomDataSampler'), 0)

        self.gl.glActiveTexture(self.gl.GL_TEXTURE2)
        self.colormapTexture.bind()
        self.m_program.setUniformValue(self.m_program.uniformLocation('colormapSampler'), 2)

        # This is an important check as it would freeze if the buffer is still
        # being written to
        self.m_vao.bind()
        if self.m_vbo.bind() and self.m_ibo.bind():

            self.gl.glDrawElements(
                self.gl.GL_POINTS,  # mode
                self.m_ibo.size(),  # count
                self.gl.GL_UNSIGNED_INT,  # type
                None  # element array buffer offset
            )
            self.m_ibo.release()

            ###########################################################
            # Drawing Selection
            if self.selected and self.selected_ibo.bind():

                self.m_program.setUniformValue(self.m_program.uniformLocation('radiusScale'), self.radiusScale * 1.15)
                self.m_program.setUniformValue(self.m_program.uniformLocation('selected'), True)
                self.gl.glDrawElements(
                    self.gl.GL_POINTS,  # mode
                    self.m_ibo.size(),  # count
                    self.gl.GL_UNSIGNED_INT,  # type
                    None  # element array buffer offset
                )
                self.selected_ibo.release()
        else:
            print('Cannot bind VBO for rendering!')

        self.m_vbo.release()
        self.m_vao.release()

        # self.m_texture.release()
        self.m_program.release()

    def display(self, camera):

        ###########################################################
        # Linear algebra
        modelMatrix = QMatrix4x4()
        viewMatrix = camera.view_matrix()
        mv = viewMatrix * modelMatrix
        mvp = camera.projectionMatrix * viewMatrix * modelMatrix
        imvp = mvp.inverted()[0]

        normalMatrix = mv.normalMatrix()

        self.draw_thermal(mv, mvp, imvp, normalMatrix, camera.projectionMatrix)

    #
    # Stuff specific to Thermals
    #
    def set_selected(self, idxs):

        if idxs is None:
            self.selected = False
            return

        self.selected = True
        index_data = (GLuint * len(idxs))(*idxs)
        self.selected_ibo.bind()
        self.m_ibo.allocate(index_data, len(idxs) * 4)      # sizeof uint
        self.selected_ibo.release()

    def update_thermal_data(self, positions):

        # self.parent.makeCurrent()

        self.selected = False
        self.buff_len = 0

        pos = positions
        k = pos.shape[0]
        vertices = pos.reshape(positions.size)

        self.m_vao.bind()
        vertex_data = (GLfloat * len(vertices))(*vertices)
        if self.m_vbo.bind():
            self.m_vbo.allocate(vertex_data, len(vertices) * 4)  # size of float
        else:
            print('Cannot bind VBO for thermal position update!')
        self.m_vbo.release()

        index_data = (GLuint * k)(*list(range(k)))
        if self.m_ibo.bind():
            self.m_ibo.allocate(index_data, k * 4)               # sizeof uint
        else:
            print('Cannot bind IBO for thermal position update!')
        self.m_ibo.release()
        self.m_vao.release()

        self.buff_len = len(positions)
