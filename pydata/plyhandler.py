from vtk import vtkPLYReader, vtkPLYWriter
from .vtkhandler import VTKHandler

class PLYHandler(VTKHandler):

    _reader_ = vtkPLYReader
    _writer_ = vtkPLYWriter
