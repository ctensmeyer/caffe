
import caffe
import numpy as np
import scipy.spatial
import polygons as poly 
import sys


def arr_to_ltuples(arr):
	points = arr.reshape((-1, 2))
	l = list()
	for idx in xrange(points.shape[0]):
		l.append( (float(points[idx][0]), float(points[idx][1])) )
	return l
	

#def package(arr):
#	points = arr.reshape((-1, 2))
#	hull = scipy.spatial.ConvexHull(points)
#	ordering = hull.vertices.tolist()
#
#	l_ordered = list()
#	for idx in ordering:
#		l_ordered.append( (points[idx][0], points[idx][1]) )
#	return l_ordered, ordering

def adaptive_mse_norm(arr1, arr2):
	arr1_x_range = np.max(arr1[::2]) - np.min(arr1[::2])
	arr2_x_range = np.max(arr2[::2]) - np.min(arr2[::2])
	arr1_y_range = np.max(arr1[1::2]) - np.min(arr1[1::2])
	arr2_y_range = np.max(arr2[1::2]) - np.min(arr2[1::2])
	norm = max(arr1_x_range, arr2_x_range) + max(arr1_y_range, arr2_y_range) / 2.
	return norm


def reorder(l_tuples):
	ordering = poly.get_polygon_order(l_tuples)
	l_ordered = list()
	for idx in ordering:
		l_ordered.append(l_tuples[idx])
	return l_ordered, ordering


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
		self.adaptive_mse_norm = False
		self.adaptive_area_norm = False

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
		num_vertex_pred = bottom[0].data.shape[1] / 2 
		num_vertex_gt = bottom[1].data.shape[1] / 2 
		count = 0
		for idx in xrange(bottom[0].data.shape[0]):
			pred, gt = bottom[0].data[idx,:], bottom[1].data[idx,:]
			pred_polygon, gt_polygon = arr_to_ltuples(pred), arr_to_ltuples(gt)
			if poly.is_poly_convex_unordered(pred_polygon) and poly.is_poly_convex_unordered(gt_polygon):
				# main expected case

				pred_polygon, _ = reorder(pred_polygon)
				gt_polygon, _ = reorder(gt_polygon)

				norm = poly.area(gt_polygon) if self.adaptive_area_norm else 1.
				p_g_area = poly.area_subtract_polygons(pred_polygon, gt_polygon)
				g_p_area = poly.area_subtract_polygons(gt_polygon, pred_polygon)
				#print gt_area, p_g_area, g_p_area

				total_p_g_area += p_g_area / norm
				total_g_p_area += g_p_area / norm
				count += 1
			else:
				# polygon vertices are not assumed to be ordered
				# therefore, if the set of points forms a non-convex polygon,
				# then there may be several orderings that correspond to distinct polygons
				if num_vertex_pred == num_vertex_gt:
					# Do (normalized) MSE for corresponding vertices
					norm = adaptive_mse_norm(pred, gt) if self.adaptive_mse_norm else 1.

					mse = 0.5 * np.sum(((pred - gt) / norm) ** 2)
					total_g_p_area += mse
					total_p_g_area += mse
					count += 1
				else:
					# vertices cannot correspond, so it's hard to do something reasonable here
					pass

		top[0].data[0] = total_g_p_area / count if count else 0.0
		top[1].data[0] = total_p_g_area / count if count else 0.0
		self.last_count = count


	def backward(self, top, propagate_down, bottom):
		bottom[0].diff[:] = 0.
		bottom[1].diff[:] = 0.
		g_p_mult = top[0].diff[0]
		p_g_mult = top[1].diff[0]
		num_vertex_pred = bottom[0].data.shape[1] / 2 
		num_vertex_gt = bottom[1].data.shape[1] / 2 
		if any(propagate_down):
			for idx in xrange(bottom[0].data.shape[0]):
				pred, gt = bottom[0].data[idx,:], bottom[1].data[idx,:]
				pred_polygon, gt_polygon = arr_to_ltuples(pred), arr_to_ltuples(gt)
				#print pred_polygon
				#print gt_polygon

				if poly.is_poly_convex_unordered(pred_polygon) and poly.is_poly_convex_unordered(gt_polygon):
					# functions in polygons module assume ordered
					pred_polygon, pred_ordering = reorder(pred_polygon)
					gt_polygon, gt_ordering = reorder(gt_polygon)

					norm = poly.area(gt_polygon) if self.adaptive_area_norm else 1.
					div = norm * self.last_count

					p_g_d_pred, p_g_d_gt = poly.d_area_subtract_polygons(pred_polygon, gt_polygon)
					g_p_d_gt, g_p_d_pred = poly.d_area_subtract_polygons(gt_polygon, pred_polygon)
					#print pred_polygon
					#print gt_polygon
					#print p_g_d_pred
					#print p_g_d_gt
					#print g_p_d_pred
					#print g_p_d_gt
					#print g_p_mult, p_g_mult

					if propagate_down[0]:
						p_g_d_pred = unpackage(p_g_d_pred, pred_ordering, 2 * num_vertex_pred)
						g_p_d_pred = unpackage(g_p_d_pred, pred_ordering, 2 * num_vertex_pred)
						bottom[0].diff[idx,:] += (g_p_mult * g_p_d_pred + p_g_mult * p_g_d_pred) / div
					if propagate_down[1]:
						p_g_d_gt = unpackage(p_g_d_gt, gt_ordering, 2 * num_vertex_gt)
						g_p_d_gt = unpackage(g_p_d_gt, gt_ordering, 2 * num_vertex_gt)
						bottom[1].diff[idx,:] += (g_p_mult * g_p_d_gt + p_g_mult * p_g_d_gt) / div
				else:
					# polygon vertices are not assumed to be ordered
					# therefore, if the set of points forms a non-convex polygon,
					# then there may be several orderings that correspond to distinct polygons
					if num_vertex_pred == num_vertex_gt:
						# Do (normalized) MSE for corresponding vertices
						norm = adaptive_mse_norm(pred, gt) if self.adaptive_mse_norm else 1.

						dmse_dpred = (pred - gt) / norm
						dmse_dgt = -1. * dmse_dpred
						if propagate_down[0]:
							bottom[0].diff[idx,:] += ((g_p_mult + p_g_mult) / self.last_count) * dmse_dpred 
						if propagate_down[1]:
							bottom[1].diff[idx,:] += ((g_p_mult + p_g_mult) / self.last_count) * dmse_dgt 
					else:
						# vertices cannot correspond, so it's hard to do something reasonable here
						pass
				#print bottom[0].diff 
				#print bottom[1].diff 


class WeightedMSELossLayer(caffe.Layer):

	def setup(self, bottom, top):
		if len(bottom) != 2:
			raise Exception("Need 2 bottom blobs")
		if len(top) != 1:
			raise Exception("Need 1 top blob:")
		self.penalty = float(self.param_str)

		# 1 ->  bottom[0] > bottom[1] leads to penalty
		# -1 -> bottom[0] < bottom[1] leads to penalty
		self.pattern = np.asarray([[1, 1, -1, 1, -1, -1, 1, -1]])

	def reshape(self, bottom, top):
		if bottom[0].data.shape[0] != bottom[1].data.shape[0]:
			raise Exception("Blobs 0 and 1 need to have the same batch size: %d vs %d" 
				% (bottom[0].data.shape[0], bottom[1].data.shape[0]))

		if bottom[0].data.shape[1] != 8:
			raise Exception("Must be shape 8")

		top[0].reshape(1)

	def forward(self, bottom, top):
		diff = (bottom[0].data - bottom[1].data) 
		coeffs = np.ones_like(diff)
		coeffs[diff * self.pattern > 0] = self.penalty
		top[0].data[0] = 0.5 * np.sum(coeffs * (diff * diff)) / bottom[0].data.shape[0]
			
	def backward(self, top, propagate_down, bottom):
		diff = (bottom[0].data - bottom[1].data)
		coeffs = np.ones_like(diff)
		coeffs[diff * self.pattern > 0] = self.penalty
		if propagate_down[0]:
			bottom[0].diff[...] = top[0].diff[0] * diff * coeffs / bottom[0].data.shape[0]
		if propagate_down[1]:
			bottom[1].diff[...] = -1 * top[0].diff[0] * diff * coeffs / bottom[0].data.shape[0]
		
		

