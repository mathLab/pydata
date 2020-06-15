"""
    Module for .VTU files

    .VTU files are VTK files with XML syntax containing vtkUnstructuredGrid.
    Further information related with the file format available at url:
    https://www.vtk.org/VTK/img/file-formats.pdf
    
"""
from vtk import vtkXMLUnstructuredGridReader, vtkXMLUnstructuredGridWriter
from vtk import vtkUnstructuredGrid, vtkPoints, vtkCellArray
from vtk import VTK_TETRA

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from .vtkhandler import VTKHandler


class VTUHandler(VTKHandler):
    """
    Handler for .VTU files.
    """

    _reader_ = vtkXMLUnstructuredGridReader
    _writer_ = vtkXMLUnstructuredGridWriter

    @classmethod
    def _polydata_from_file(cls, filename):
        """
        Private method to extract vtkTetraData from `filename`. The `filename`
        have to be well-formatted VTU file.

        :param str filename: the name of the file to parse.
        :return: the dataset
        :rtype: vtkTetraData
        """
        reader = cls._reader_()
        reader.SetFileName(filename)
        reader.Update()

        return reader.GetOutput()

    @classmethod
    def write(cls, filename, data):
        """
        Method to save the dataset to `filename`. The dataset `data` should be
        a dictionary containing the requested information. The obtained
        `filename` is a well-formatted VTU file.

        :param str filename: the name of the file to write.
        :param dict data: the dataset to save.

        .. warning:: all the cells will be stored as VTK_TETRA.
            
        """

        unstructured_grid = vtkUnstructuredGrid()

        points = vtkPoints()
        points.SetData(numpy_to_vtk(data['points']))

        cells = vtkCellArray()
        for cell in data['cells']:
            cells.InsertNextCell(len(cell), cell)

        if 'point_data' in data:
            for name, array in data['point_data'].items():
                vtu_array = numpy_to_vtk(array)
                vtu_array.SetName(name)
                unstructured_grid.GetPointData().AddArray(vtu_array)

        if 'cell_data' in data:
            for name, array in data['cell_data'].items():
                vtu_array = numpy_to_vtk(array)
                vtu_array.SetName(name)
                unstructured_grid.GetCellData().AddArray(vtu_array)

        unstructured_grid.SetPoints(points)
        unstructured_grid.SetCells(VTK_TETRA, cells)

        writer = cls._writer_()
        writer.SetFileName(filename)
        writer.SetInputData(unstructured_grid)
        writer.Write()
