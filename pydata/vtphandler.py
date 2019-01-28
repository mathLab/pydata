from vtk import vtkXMLPolyDataReader, vtkXMLPolyDataWriter
from .vtkhandler import VTKHandler


class VTPHandler(VTKHandler):

    _reader_ = vtkXMLPolyDataReader
    _writer_ = vtkXMLPolyDataWriter
