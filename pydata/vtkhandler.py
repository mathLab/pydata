from vtk import vtkDataSetReader, vtkDataSetWriter, vtkPolyData, vtkPoints, vtkCellArray
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import numpy as np

class VTKHandler(object):

    _reader_ = vtkDataSetReader
    _writer_ = vtkDataSetWriter

    @classmethod
    def _polydata(cls, filename):
        reader = cls._reader_()
        reader.SetFileName(filename)
        reader.Update()
        return reader.GetOutput()

    @classmethod
    def read(cls, filename):

        data = cls._polydata(filename)
        result = {
            'cells': [],
            'points': None
        }
        for id_cell in range(data.GetNumberOfCells()):
            cell = data.GetCell(id_cell)
            result['cells'].append([
                cell.GetPointId(id_point) 
                for id_point in range(cell.GetNumberOfPoints())]
            )

        result['points'] = vtk_to_numpy(data.GetPoints().GetData())

        return result

    @classmethod
    def write(cls, filename, data):
        
        polydata = vtkPolyData()

        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(data.points))

        vtk_cells = vtkCellArray()
        for cell in data.cells:
            vtk_cells.InsertNextCell(len(cell), cell)

        polydata.SetPoints(vtk_points)
        polydata.SetPolys(vtk_cells)

        writer = cls._writer_()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()
