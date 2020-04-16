from vtk import vtkPolyDataReader, vtkPolyDataWriter
from vtk import vtkPolyData, vtkPoints, vtkCellArray

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import numpy as np

class VTKHandler(object):
    """ TODO """
    _reader_ = vtkPolyDataReader
    _writer_ = vtkPolyDataWriter

    @classmethod
    def _polydata_from_file(cls, filename):
        """
        Load a vtkPolyData object from `filename`.

        :param str filename: the name of the file where data is stored.
        :return: geometrical and topological information
        :rtype: vtk.vtkPolyData
        """
        reader = cls._reader_()
        reader.SetFileName(filename)
        reader.Update()
        return reader.GetOutput()

    @classmethod
    def _polydata_to_file(cls, filename, polydata):
        """
        Load a vtkPolyData object from `filename`.

        :param str filename: the name of the file where data is stored.
        :return: geometrical and topological information
        :rtype: vtk.vtkPolyData
        """
        writer = cls._writer_()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()

    @classmethod
    def read(cls, filename):

        data = cls._polydata_from_file(filename)
        result = {'cells': [], 'points': None}

        for id_cell in range(data.GetNumberOfCells()):
            cell = data.GetCell(id_cell)
            result['cells'].append([
                cell.GetPointId(id_point)
                for id_point in range(cell.GetNumberOfPoints())
            ])

        result['points'] = vtk_to_numpy(data.GetPoints().GetData())

        # Point data
        for i in range(data.GetPointData().GetNumberOfArrays()):
            array = vtk_to_numpy(data.GetPointData().GetArray(i))
            name = data.GetPointData().GetArrayName(i)
            if 'point_data' not in result:
                result['point_data'] = dict()
            result['point_data'][name] = array

        # Cell data
        for i in range(data.GetCellData().GetNumberOfArrays()):
            array = vtk_to_numpy(data.GetCellData().GetArray(i))
            name = data.GetCellData().GetArrayName(i)
            if 'cell_data' not in result:
                result['cell_data'] = dict()
            result['cell_data'][name] = array

        return result

    @classmethod
    def write(cls, filename, data):
        """ TODO """
        polydata = vtkPolyData()

        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(data['points']))

        vtk_cells = vtkCellArray()
        for cell in data['cells']:
            vtk_cells.InsertNextCell(len(cell), cell)

        if 'point_data' in data:
            for name, array in data['point_data'].items():
                vtk_array = numpy_to_vtk(array)
                vtk_array.SetName(name)
                polydata.GetPointData().AddArray(vtk_array)

        if 'cell_data' in data:
            for name, array in data['cell_data'].items():
                vtk_array = numpy_to_vtk(array)
                vtk_array.SetName(name)
                polydata.GetCellData().AddArray(vtk_array)

        polydata.SetPoints(vtk_points)
        polydata.SetPolys(vtk_cells)

        cls._polydata_to_file(filename, polydata)
