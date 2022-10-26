import numpy as np
import matplotlib.pyplot as plt
import os

from OpenGL.raw.GL._types import GLfloat, GLuint, GLbyte

from PyQt5.QtGui import (
        QOpenGLBuffer,
        QOpenGLShader,
        QOpenGLShaderProgram,
        QOpenGLVersionProfile,
        QOpenGLVertexArrayObject,
        QSurfaceFormat,
        QOpenGLTexture,
    )
from glidar_analyst.util.resource_manager import ResourceManager


class BaseRenderer:

    def __init__(self, gl):

        self.gl = gl

        self.surface_program = None

    def closeEvent(self):
        raise NotImplementedError()

    def display(self, camera):

        # TODO: Figure out a nice way to pass the MVP data to each renderer
        raise NotImplementedError()

    def keyPressedEvent(self, event):
        pass

    #
    # Utility functions
    @staticmethod
    def read_file(name):
        res = None
        with open(name, 'r') as f:
            res = f.read()
        return res

    @staticmethod
    def initTexture(texture, data):

        data = data.reshape(data.size)
        pixels = (GLfloat * data.size)(*data)

        texture.setFormat(QOpenGLTexture.RGB32F)
        texture.setWrapMode(
            QOpenGLTexture.DirectionS,
            QOpenGLTexture.ClampToEdge
            )
        texture.setSize(len(data)/4)
        texture.allocateStorage()
        texture.setData(
            0,  # MIP level
            0,  # layer
            QOpenGLTexture.CubeMapPositiveX,    # Hopefully unused
            QOpenGLTexture.RGBA,                # Pixel format
            QOpenGLTexture.Float32,             # Pixel type
            pixels
        )

    def createShadersFromFolder(self, folder):

        folder = ResourceManager().get_absolute_path(folder)

        self.surface_program = QOpenGLShaderProgram()

        ###########################################################
        # VERTEX SHADER
        if self.surface_program.addShaderFromSourceCode(
                QOpenGLShader.Vertex, self.read_file(folder + 'vert.glsl')):
            print('Vertex shader compiled successfully.')
        else:
            print('Error compiling vertex shader.')
            print(self.surface_program.log())
            raise RuntimeError('Vertex shader.')

        ###########################################################
        # GEOMETRY SHADER
        if os.path.exists(folder + 'geom.glsl'):
            if self.surface_program.addShaderFromSourceCode(
                    QOpenGLShader.Geometry, self.read_file(folder + 'geom.glsl')):
                print('Geometry shader compiled successfully.')
            else:
                print('Error compiling geometry shader.')
                print(self.surface_program.log())
                raise RuntimeError('Geometry shader.')

        ###########################################################
        # FRAGMENT SHADER
        if self.surface_program.addShaderFromSourceCode(
                QOpenGLShader.Fragment, self.read_file(folder + 'frag.glsl')):
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