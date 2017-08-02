
import sys
import unittest
import numpy as np
from test_gradient_for_python_layer import test_gradient_for_python_layer

def helper(pred, gt):
	input_names_and_values = [('pred', pred), 
							  ('gt', gt)]
	output_names = ['g_p_area', 'p_g_area']
	py_module = 'polygon_area_layer'
	py_layer = 'PolygonDifferenceAreaLayer'
	param_str = ""
	propagate_down = [True, True]
	test_gradient_for_python_layer(input_names_and_values, output_names, 
								py_module, py_layer, param_str, propagate_down)

class PolygonDifferenceAreaTest(unittest.TestCase):

	def test_simple_contains(self):
		pred = np.asarray([[0.1, 0.2,
						    0.7, 0.1,
						    0.8, 0.8,
						    0.2, 0.9]])
		gt = np.asarray([[0.0, 0.0,
						  1.0, 0.0,
						  1.0, 1.0,
						  0.0, 1.0]])
		helper(pred, gt)

	def test_simple_contains2(self):
		gt = np.asarray([[0.1, 0.1,
						  0.7, 0.2,
						  0.9, 0.8,
						  0.3, 0.9]])
		pred = np.asarray([[0.0, 0.0,
						    1.0, 0.0,
						    1.0, 1.0,
						    0.0, 1.0]])
		helper(pred, gt)

	def test_non_overlap(self):
		gt = np.asarray([[0.1, 0.1,
						  0.7, 0.2,
						  0.9, 0.8,
						  0.3, 0.9]])
		pred = np.asarray([[1.0, 1.0,
						    2.0, 1.0,
						    2.0, 2.0,
						    1.0, 2.0]])
		helper(pred, gt)

	def test_single_overlap(self):
		pred = np.asarray([[0.3, 0.3,
						    1.3, 0.3,
						    1.3, 0.6,
						    0.3, 0.6]])
		gt = np.asarray([[0.0, 0.0,
						  1.0, 0.0,
						  1.1, 1.0,
						  0.1, 1.0]])
		helper(pred, gt)

	def test_multiple_overlap(self):
		pred = np.asarray([[-0.3, 0.3,
						    1.3, 0.3,
						    1.3, 0.6,
						    -0.3, 0.6]])
		gt = np.asarray([[0.0, 0.0,
						  1.0, 0.0,
						  1.1, 1.0,
						  0.1, 1.0]])
		helper(pred, gt)

	def test_multiple_overlap2(self):
		pred = np.asarray([[-0.3, -0.3,
						    2.0, 0.5,
						    1.3, 1.6,
						    -0.1, 0.2]])
		gt = np.asarray([[0.0, 0.0,
						  1.0, 0.1,
						  1.1, 1.0,
						  0.1, 1.1]])
		helper(pred, gt)

	def test_multiple_overlap2_reorder(self):
		pred = np.asarray([[-0.3, -0.3,
						    1.3, 1.6,
						    2.0, 0.5,
						    -0.1, 0.2]])
		gt = np.asarray([[1.0, 0.1,
						  1.1, 1.0,
						  0.0, 0.0,
						  0.1, 1.1]])
		helper(pred, gt)

	def test_nonconvex(self):
		pred = np.asarray([[0.5, 0.5,
						    0.5, 0.2,
						    0.2, 0.8,
						    0.8, 0.7]])
		gt = np.asarray([[0.0, 0.0,
						  1.0, 0.0,
						  1.0, 1.0,
						  0.0, 1.0]])
		helper(pred, gt)

	def test_nonconvex_noncorrepsonding(self):
		pred = np.asarray([[0.5, 0.5,
						    0.5, 0.2,
						    0.2, 0.8,
						    0.8, 0.7,
							0.2, 0.2]])
		gt = np.asarray([[0.0, 0.0,
						  1.0, 0.0,
						  1.0, 1.0,
						  0.0, 1.0]])
		helper(pred, gt)

	def test_batch(self):
		pred = np.asarray([[0.5, 0.5,
						    0.5, 0.2,
						    0.2, 0.8,
						    0.8, 0.7],
						   [-0.3, -0.3,
						    1.3, 1.6,
						    2.0, 0.5,
						    -0.1, 0.2]
						   ])
		gt = np.asarray([[0.0, 0.0,
						  1.0, 0.0,
						  1.0, 1.0,
						  0.0, 1.0],
						 [1.0, 0.1,
						  1.1, 1.0,
						  0.0, 0.0,
						  0.1, 1.1]
						 ])
		
		
if __name__ == "__main__":
	unittest.main()

