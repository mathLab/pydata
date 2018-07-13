from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os

from pydata.vtphandler import VTPHandler

vtp_file = 'tests/test_datasets/cube.vtp'

class TestVTPHandler(TestCase):
    def test_points(self):
        data = VTPHandler.read(vtp_file)
        np.testing.assert_array_almost_equal(data['points'][0], [-0.5]*3)

    def test_number_points(self):
        data = VTPHandler.read(vtp_file)
        np.testing.assert_equal(data['points'].shape, (24, 3))

    def test_cells(self):
        data = VTPHandler.read(vtp_file)
        np.testing.assert_equal(data['cells'][5], [20, 21, 23, 22])

