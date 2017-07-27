
import caffe
import numpy as np
import scipy.spatial
from polygons import *


def package(arr):
	points = arr.reshape((-1, 2))
	hull = scipy.spatial.ConvexHull(points)
	ordering = hull.vertices.tolist()

	l_ordered = list()
	for idx in ordering:
		l_ordered.append( (points[idx][0], points[idx][1]) )
	return l_ordered, ordering

#def package(arr):
#	l = list()
#	for idx in xrange(0, arr.shape[0], 2):
#		l.append( (arr[idx], arr[idx+1]) )
#	ordering = get_polygon_order(l)
#	l_ordered = list()
#	for idx in ordering:
#		l_ordered.append(l[idx])
#	return l_ordered, ordering


def unpackage(l_tuples, ordering, original_len):
	arr = np.zeros(shape=(original_len,) )
	for idx, idx2 in enumerate(ordering):
		arr[2*idx2] = l_tuples[idx][0]
		arr[2*idx2+1] = l_tuples[idx][1]
	return arr


#if __name__ == "__main__":
#	ordering = [0, 2, 1, 3]
#	arr = np.arange(8)
#	packaged = package(arr, ordering)
#	print packaged
#	unpackaged = unpackage(packaged, ordering)
#	print unpackaged
#	print arr

class PolygonDifferenceAreaLayer(caffe.Layer):
	'''
	Caffe Layer with 
	2 input blobs:
		0) Predicted Coords
		1) GT Coords
	2 outputs:
		0) Area(G - P) / Area(G)
		1) Area(P - G) / Area(G)
	'''
	def setup(self, bottom, top):
		if len(bottom) != 2:
			raise Exception("Need 2 bottom blobs: pred_coords, gt_coords")
		if len(top) != 2:
			raise Exception("Need 2 top blobs: A(G-P)/A(G), A(P-G)/A(G)")

	def reshape(self, bottom, top):
		if bottom[0].data.shape[0] != bottom[1].data.shape[0]:
			raise Exception("Blobs 0 and 1 need to have the same batch size: %d vs %d" 
				% (bottom[0].data.shape[0], bottom[1].data.shape[0]))
		for idx in [0,1]:
			if bottom[idx].data.shape[1] % 2 == 1:
				raise Exception("Bottom blob %d axis 1 must be even: %d" % (idx, bottom[0].data.shape[0]))

		# scalar outputs
		top[0].reshape(1)
		top[1].reshape(1)


	def forward(self, bottom, top):
		total_p_g_area = 0.
		total_g_p_area = 0.
		for idx in xrange(bottom[0].data.shape[0]):
			pred_polygon, _ = package(bottom[0].data[idx,:])
			gt_polygon, _ = package(bottom[1].data[idx,:])

			gt_area = area(gt_polygon)
			p_g_area = area_subtract_polygons(pred_polygon, gt_polygon)
			g_p_area = area_subtract_polygons(gt_polygon, pred_polygon)

			total_p_g_area += p_g_area / gt_area
			total_g_p_area += g_p_area / gt_area

		top[0].data[0] = total_g_p_area / bottom[0].data.shape[0]
		top[1].data[0] = total_p_g_area / bottom[0].data.shape[0]


	def backward(self, top, propagate_down, bottom):
		bottom[0].diff[:] = 0.
		bottom[1].diff[:] = 0.
		g_p_mult = top[0].diff[0]
		p_g_mult = top[1].diff[0]
		pred_len = bottom[0].data.shape[1]
		gt_len = bottom[1].data.shape[1]
		if any(propagate_down):
			for idx in xrange(bottom[0].data.shape[0]):
				pred_polygon, pred_ordering = package(bottom[0].data[idx,:])
				gt_polygon, gt_ordering = package(bottom[1].data[idx,:])

				gt_area = area(gt_polygon)
				div = gt_area * bottom[0].data.shape[0]

				p_g_d_pred, p_g_d_gt = d_area_subtract_polygons(pred_polygon, gt_polygon)
				g_p_d_pred, g_p_d_gt = d_area_subtract_polygons(pred_polygon, gt_polygon)

				if propagate_down[0]:
					p_g_d_pred = unpackage(p_g_d_pred, pred_ordering, pred_len)
					g_p_d_pred = unpackage(g_p_d_pred, pred_ordering, pred_len)
					bottom[0].diff[idx,:] += (g_p_mult * g_p_d_pred + p_g_mult * p_g_d_pred) / div
				if propagate_down[1]:
					p_g_d_gt = unpackage(p_g_d_gt, gt_ordering, gt_len)
					g_p_d_gt = unpackage(g_p_d_gt, gt_ordering, gt_len)
					bottom[1].diff[idx,:] += (g_p_mult * g_p_d_gt + p_g_mult * p_g_d_gt) / div
			



