
import caffe
import numpy as np
from adaptive_gt_raw import adaptive_gt

class AdaptiveBaselineGTLayer(caffe.Layer):
	'''
	Caffe Layer with 
	3 input blobs:
		0) Predicted Probabilities
		#1) GT baselines.  Assumed to be 1D connected components with
		#	0 as background and 1 as foreground
		2) Precomputed GT distance maps/threshold regions
	3 outputs:
		0) Adapted Binary GT
	Parameters:
		Vertical Penalty - Multiplicative constant for vertical distance from GT line as a term in the
			energy cost of the DP for seam carving
		Horizontal Penalty - Multiplicative constant for signed horizontal distance GT start/end points as
			a term in the energy cost of the DP for seam carving
	'''

	def setup(self, bottom, top):
		if len(bottom) != 2:
			raise Exception("Needs 3 bottom blobs: Probs, GT, Precomputed Distance Map")
		if len(top) != 1:
			raise Exception("Only 1 top blob: Adaptive GT")
		self.vert_penalty = 0.000
		self.horz_penalty = 0.5
		self.tolerance = 20

	def reshape(self, bottom, top):
		#if bottom[0].count != bottom[1].count:
		#	raise Exception("Blobs 0 and 1 need to have the same count: %d vs %d" % (bottom[0].count, bottom[1].count))
		if 3 * bottom[0].count != bottom[1].count:
			raise Exception("Blob 1 must have 3x the count of Blob 0: %d vs %d" % (bottom[1].count, 3 * bottom[0].count))

		s = bottom[0].data.shape
		top[0].reshape(*s)
			

	def forward(self, bottom, top):
		for idx in xrange(top[0].data.shape[0]):
			top[0].data[idx,:,:,:] = self.curtis_code(bottom[0].data[idx,0,:,:], bottom[1].data[idx,:,:,:],
				self.tolerance, self.vert_penalty, self.horz_penalty)

	def curtis_code(self, predicted_probs, precomputed, tolerance, vert_penalty, horz_penalty):
		'''
		predicted_probs HxW numpy array (np.float32)
		original_gt HxW numpy array of binary values (np.float32)
		precomputed 3xHxW numpy array (np.float32)
		vert_penalty - float - penalty/pixel away from GT baseline
		horz_penalty - float - not used
		'''
		return adaptive_gt(predicted_probs, precomputed, tolerance=tolerance, alpha=vert_penalty, beta=1.414)


	def backward(self, top, propagate_down, bottom):
		pass
		#raise Exception("Not Implemented/Defined")

