import numpy as np
from io import BytesIO


class LSDYNAHandler(object):
    @classmethod
    def read(cls, filename):

        with open(filename, 'r') as f:
            lines = [line for line in f.readlines() if not line.startswith('$')]

        index_pts = lines.index('*NODE\n')
        index_cls = lines.index('*ELEMENT_SHELL\n')

        pts_string = BytesIO(
            str.encode(''.join(lines[index_pts + 1:index_cls])))
        pts = np.genfromtxt(pts_string)[:, 1:]

        cls_string = BytesIO(str.encode(''.join(lines[index_cls + 1:-1])))
        cls = np.genfromtxt(cls_string, dtype=int)[:, 2:] - 1

        result = {'points': pts, 'cells': cls}
        return result

    @classmethod
    def write(cls, filename, data):

        string = ''
        string += '*KEYWORD\n'
        string += '*NODE\n'
        for idpt, pt in enumerate(data.points, start=1):
            string += '{:8d} {:15.10f} {:15.10f} {:15.10f}\n'.format(
                idpt, pt[0], pt[1], pt[2])
        string += '*ELEMENT_SHELL\n'
        for idcl, cl in enumerate(data.cells, start=1):
            string += '{:8d} {:8d} {:8d} {:8d} {:8d} {:8d}\n'.format(
                idcl, 1, cl[0] + 1, cl[1] + 1, cl[2] + 1, cl[3] + 1)
        string += '*END\n'

        with open(filename, 'w') as f:
            f.write(string)
