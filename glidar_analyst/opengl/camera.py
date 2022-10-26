from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QVector3D, QMatrix4x4

import numpy


class Camera:

    def __init__(self, parent):

        self.parent = parent

        self.projectionMatrix = QMatrix4x4()
        self.camera_position = QVector3D()

        self.rotateX = 0
        self.rotateY = 0
        self.moveX = 0
        self.moveY = 0
        self.scale = 1

        self.mousePosition = None

    def view_matrix(self):

        viewMatrix = QMatrix4x4()

        # Third Zoom in or out
        viewMatrix.translate(QVector3D(0, 0, -10))  # -10 in the camera z direction
        viewMatrix.scale(self.scale)
        # viewMatrix.scale(100)

        # Second make it look where we want to
        viewMatrix.rotate(self.rotateY, QVector3D(1, 0, 0))
        viewMatrix.rotate(self.rotateX, QVector3D(0, 0, 1))

        # First move the camera to the correct place
        viewMatrix.translate(self.moveX, self.moveY, 0)

        return viewMatrix

    def mousePressEvent(self, event):
        self.mousePosition = event.pos()

    def mouseMoveEvent(self, event):
        pos = event.pos()

        if int(Qt.LeftButton) & int(event.buttons()):
            self.rotateX += (self.mousePosition.x() - pos.x()) / 10.
            self.rotateY += (self.mousePosition.y() - pos.y()) / 10.
            if self.rotateY > 90:
                self.rotateY = 90
            if self.rotateY < -90:
                self.rotateY = -90

        elif int(Qt.RightButton) & int(event.buttons()):

            middle = QPoint(self.parent.width() // 2, self.parent.height() // 2)
            l1 = self.qPoint_distance(middle, pos)
            l2 = self.qPoint_distance(middle, self.mousePosition)
            self.scale *= numpy.exp((l1 - l2) / 200)

        elif int(Qt.MiddleButton) & int(event.buttons()):
            s = numpy.sin(self.rotateX * numpy.pi / 180)
            c = numpy.cos(self.rotateX * numpy.pi / 180)

            dx = 0.02 * (self.mousePosition.x() - pos.x()) / self.scale
            dy = 0.02 * (self.mousePosition.y() - pos.y()) / self.scale

            self.moveX -= dx * c - dy * s
            self.moveY += c * dy + s * dx

        self.mousePosition = pos

    def qPoint_distance(self, p, q):
        v = p - q
        return numpy.sqrt(v.x()**2 + v.y()**2)

    def lookAt(self, x, y):

        s = numpy.sin(self.rotateX * numpy.pi / 180)
        c = numpy.cos(self.rotateX * numpy.pi / 180)

        self.moveX = - x + 100 * s
        self.moveY = - y - 100 * c
