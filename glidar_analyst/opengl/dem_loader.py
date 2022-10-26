import os
import numpy as np
import pandas as pd

# from isotrotter.opengl.dem_loader import make_triangle_strip
#


def convert_file(filename):

    f = open(filename)

    line = f.read(1024)
    Data={
            "Descriptive Name of the represented area                                 ": (  0, 134),
            "nothing                                                                  ": (150, 155),
            "?                                                                        ": (156, 161),
            "UTM Zone number                                                          ": (162, 167),
            "Unit of resolution of ground grid (0=radian;1=feet;2=metre;3=arc-second) ": (529, 534),
            "Unit of resolution Elevation (1=feet;2=metre)                            ": (535, 540),
            "Easting of the South West corner                                         ": (546, 569),
            "Northing of the South West corner                                        ": (570, 593),
            "Easting of the North West corner                                         ": (594, 617),
            "Northing of the North West corner                                        ": (618, 641),
            "Easting of the North East corner                                         ": (642, 665),
            "Northing of the North East corner                                        ": (666, 689),
            "Easting of the South East corner                                         ": (690, 713),
            "Northing of the South East corner                                        ": (714, 737),
            "Minimum elevation found in this file                                     ": (738, 761),
            "Maximum elevation found in this file                                     ": (762, 786),
            "Resolution per grid cell East – West                                     ": (816, 827),
            "Resolution per grid cell North – South                                   ": (828, 839),
            "Number of columns                                                        ": (858, 864)}

    x0 = float(line[546:569])
    y0 = float(line[570:593])

    x1 = float(line[642:665])
    y1 = float(line[666:689])

    for k in Data:
        print(k, line[Data[k][0]:Data[k][1]])

    def chunkstring(string, length):
        return (string[0+i:length+i] for i in range(0, len(string), length))

    def parse_column(start):
        line = f.read(1024)

        b = np.array(list(chunkstring(line[start+144:start+1020], 6)), dtype=int)

        blocks = [np.array(list(chunkstring(f.read(1024).rstrip(), 6)), dtype=int) 
                    for _ in range(29)]     # TODO: Figure out the number of pages

        blocks = [b] + blocks

        column = np.concatenate(blocks)
        return column

    data = [parse_column(0) for i in range(5041)]
    grid = np.stack(data)

    f.close()
    
    return grid, x0, y0, x1, y1


def make_triangle_strip(grid, x0, y0, x1, y1, step):

    vertices = []
    xcoords = np.linspace(x0, x1, grid.shape[0]) - x0
    ycoords = np.linspace(y0, y1, grid.shape[1]) - y0

    print(grid.shape)

    for i in range(0, grid.shape[0], step)[:-1]:

        if (i//step) % 2 == 0:
            rng = range(0, grid.shape[1], step)
            for j in rng:
                # Right column
                vertices.append(xcoords[i+step])
                vertices.append(ycoords[j])
                vertices.append(grid[i+step,j]/10.)

                # Left column
                vertices.append(xcoords[i])
                vertices.append(ycoords[j])
                vertices.append(grid[i,j]/10.)

            # if len(list(rng)) % 2 == 1:
            #     print('adding extra point at the bottom')
            #     # Right column
            #     vertices.append(xcoords[i+step])
            #     vertices.append(ycoords[j])
            #     vertices.append(grid[i+step,j]/10.)

        else:
            rng = reversed(range(0, grid.shape[1], step))
            for j in rng:
                # Left column
                vertices.append(xcoords[i])
                vertices.append(ycoords[j])
                vertices.append(grid[i,j]/10.)

                # Right column
                vertices.append(xcoords[i+step])
                vertices.append(ycoords[j])
                vertices.append(grid[i+step,j]/10.)

    return vertices, x0, y0, len(xcoords)//step +1, len(ycoords)//step + 1


def compute_normals(strip, height):

    height *= 2         # There is twice as many triangles as points in a single column
    print(strip.shape)

    vertices = strip
    vertices.shape = (strip.size // 3, 3)
    print('vertices', vertices.size, vertices.shape)

    t0, t1, t2 = None, vertices[0], vertices[1],

    normals = []
    for i, v in enumerate(vertices[2:]):

        t0 = t1
        t1 = t2
        t2 = v

        e0 = t0 - t1
        e1 = t2 - t1

        n = np.cross(e0, e1)

        if i % 2 == 0:
            n *= -1.0
        normals.append(n)

    normals = np.array(normals)
    print('normals', normals.shape)

    result = []

    # index_array = dict()
    for i in range(normals.shape[0]):

        column = i // height
        row = i % height

        # forward = (column % 2 == row % 2)
        forward = ((i % 2) == (column % 2))

        if forward:
            buddy = i + 2 * (height - row - 1)
        else:
            buddy = i - 2 * (row + 1)

        idx = []
        if buddy > 2 and buddy < normals.shape[0]:
            idx += [buddy, buddy - 1, buddy - 2]

        if i > 2 and i < normals.shape[0]:
            idx += [i, i - 1, i - 2]

        if i == 1:
            idx = [0, 1]

        # index_array[i] = idx

        n = np.zeros(3)
        for j in idx:
            n += normals[j]

        n /= np.sqrt(np.sum(n**2))
        result.append(n)

    result.append(np.array([0.,0.,-1.]))
    result.append(np.array([0.,0.,-1.]))

    # print(index_array)

    # print(result)

    return np.array(result), None


def load_dem_surface(file, step=10):

    vert, x0, y0, w, h, = make_triangle_strip(*convert_file(file), step)
    normals, ia = compute_normals(np.array(vert), h)

    return (vert, x0, y0), normals.flatten()


def write_obj(grid, x0, y0, x1, y1, xx, yy, step, filename):
    
    xcoords = np.linspace(x0, x1, grid.shape[0])
    
    if xx is None:
        xcoords = xcoords - xcoords[0]
    else:
        xcoords = xcoords - xx
    
    ycoords = np.linspace(y0, y1, grid.shape[1])
    if yy is None:
        ycoords = ycoords - ycoords[0]
    else:
        ycoords = ycoords - yy
    
    file = open(filename[:-3]+'obj',"w+")

    step = 10
    file.write('o Surface\n')

    for y in range(0, grid.shape[1], step):
        for x in range(0, grid.shape[0], step):
            file.write('v {:.6f} {:.6f} {:.6f}\n'.format(xcoords[x], ycoords[y], float(grid[x,y])/10.))

    N = len(range(0, grid.shape[0], step))

    def idx(i, j):
        return i + j * N + 1       

    for j in range(0, grid.shape[1]//step):        
        for i in range(0, grid.shape[0]//step):
            file.write('f {} {} {}\n'.format(idx(i, j), idx(i, j+1), idx(i+1, j)))
            file.write('f {} {} {}\n'.format(idx(i+1, j), idx(i, j+1), idx(i+1, j+1)))              

    file.close()


if __name__ == '__main__':

    folder = '../../data/Charts/Norgeskart/DEM/'
    files = os.listdir(folder)

    strip, x0, y0, width, height = make_triangle_strip(*convert_file(folder + files[0]), 100)

    print(width, height)
    normals, ia = compute_normals(np.array(strip), height)

    print(normals)
