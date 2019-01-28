""" TODO """
import os
import importlib


class Dataset(object):
    handlers = {
        '.vtk': ('.vtkhandler', 'VTKHandler'),
        '.vtp': ('.vtphandler', 'VTPHandler'),
        '.vtu': ('.vtuhandler', 'VTUHandler'),
        '.ply': ('.plyhandler', 'PLYHandler'),
        '.stl': ('.stlhandler', 'STLHandler'),
        '.iges': ('.igeshandler', 'IGESHandler'),
        '.k': ('.lsdynahandler', 'LSDYNAHandler')
    }

    def read(self, filename):
        """ TODO """
        module, handler = self.handlers.get(os.path.splitext(filename)[1])
        reader = getattr(
            importlib.import_module(module, package='pydata'), handler)
        for attr, value in reader.read(filename).items():
            setattr(self, attr, value)

    def write(self, filename):
        """ TODO """
        module, handler = self.handlers.get(os.path.splitext(filename)[1])
        writer = getattr(
            importlib.import_module(module, package='pydata'), handler)
        writer.write(filename, vars(self))
