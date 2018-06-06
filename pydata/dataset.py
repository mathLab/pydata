import os
import importlib
import sys

class Dataset(object):
    handlers = {
        '.vtk': ('vtkhandler', 'VTKHandler'),
        '.stl': ('stlhandler', 'STLHandler'),
        '.iges': ('igeshandler', 'IGESHandler')
    }

    def read(self, filename):
        module, handler = self.handlers.get(os.path.splitext(filename)[1])
        reader = getattr(importlib.import_module(module), handler)
        for attr, value in reader.read(filename).items():
            setattr(self, attr, value)

    def write(self, filename):
        module, handler = self.handlers.get(os.path.splitext(filename)[1])
        writer = getattr(importlib.import_module(module), handler)
        writer.write(filename, self)
