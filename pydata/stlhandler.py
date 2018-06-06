from vtk import vtkSTLReader, vtkSTLWriter
from .vtkhandler import VTKHandler

class STLHandler(VTKHandler):

    _reader_ = vtkSTLReader
    _writer_ = vtkSTLWriter
