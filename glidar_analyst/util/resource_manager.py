import os

from glidar_analyst.util.singleton import Singleton


class ResourceManager(metaclass=Singleton):
    """
    This keeps track of where the hell are all the
    data stored.
    """

    def __init__(self, path):

        self.base_path = path

    def get_absolute_path(self, relative):

        return os.path.join(self.base_path, relative)


if __name__ == '__main__':

    rm1 = ResourceManager('./here')
    rm2 = ResourceManager()

    print(rm1.base_path, rm2.base_path)