from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os

from pydata import STLHandler

stl_file = 'tests/test_datasets/test_sphere.stl'


class TestStlHandler(TestCase):
    def test_points(self):
        data = STLHandler.read(stl_file)
        np.testing.assert_array_almost_equal(data['points'][0], 
                [-21.319759, -10.33176 ,  39.370079])

