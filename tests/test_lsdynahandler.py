from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os

from pydata.lsdynahandler import LSDYNAHandler

k_file = 'tests/test_datasets/cube.k'


class TestLSDYNAHandler(TestCase):
    def test_points(self):
        data = LSDYNAHandler.read(k_file)
        np.testing.assert_array_almost_equal(data['points'][0], [-0.5] * 3)

    def test_number_points(self):
        data = LSDYNAHandler.read(k_file)
        np.testing.assert_equal(data['points'].shape, (24, 3))

    def test_cellss(self):
        data = LSDYNAHandler.read(k_file)
        np.testing.assert_equal(data['cells'][5], [20, 21, 23, 22])
