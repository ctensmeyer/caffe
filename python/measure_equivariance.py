
import os
import re
import sys
import cv2
import h5py
import math
import lmdb
import caffe
import errno
import shutil
import random
import pprint
import tempfile
import argparse
import traceback
import scipy.stats
import numpy as np
import scipy.ndimage
import caffe.proto.caffe_pb2
from caffe import layers as L, params as P


def setup_scratch_space(args):
	try:
		os.makedirs("./tmp")
	except:
		pass
	#args.tmp_dir = #tempfile.mkdtemp()
	args.tmp_dir = "./tmp"
	args.train_file = os.path.join(args.tmp_dir, "train_val.prototxt")
	args.train_db = os.path.join(args.tmp_dir, "train.h5")
	args.train_db_list = os.path.join(args.tmp_dir, "train_list.txt")
	with open(args.train_db_list, 'w') as f:
		f.write('%s\n' % args.train_db)

	args.test_db = os.path.join(args.tmp_dir, "test.h5")
	args.test_db_list = os.path.join(args.tmp_dir, "test_list.txt")
	with open(args.test_db_list, 'w') as f:
		f.write('%s\n' % args.test_db)
	args.solver_file = os.path.join(args.tmp_dir, "solver.prototxt")


def cleanup_scratch_space(args):
	shutil.rmtree(args.tmp_dir)


def equivariance_proto(args, num_features, num_classes, l2_loss_weight=1, mlp=False):
	n = caffe.NetSpec()
	ce_loss_weight = 1 - l2_loss_weight

	n.input_features, n.target_features, n.target_output_probs, n.labels = L.HDF5Data(
		batch_size=args.train_batch_size, source=args.train_db_list, ntop=4, include=dict(phase=caffe.TRAIN))
	n.VAL_input_features, n.VAL_target_features, n.VAL_target_output_probs, n.VAL_labels = L.HDF5Data(
		batch_size=1, source=args.test_db_list, ntop=4, include=dict(phase=caffe.TEST))

	if mlp:
		n.prev = L.InnerProduct(n.input_features, num_output=int(args.hidden_size * num_features), name='mlp_hidden')
		n.prev = L.ReLU(n.prev, in_place=True)
	else:
		n.prev = n.input_features

	n.prev = L.InnerProduct(n.prev, num_output=num_features, name='mapping')
	n.reconstruction = L.ReLU(n.prev, name='reconstruction')  # assumes that target_features are rectified

	# caffe will automatically insert split layers when two or more layers have the same bottom, but
	# in so doing, it mangles the name.  By explicitly doing the split, we control the names so that 
	# the blob values can be accessed by the names given here
	n.reconstruction1, n.reconstruction2 = L.Split(n.reconstruction, ntop=2)  

	n.reconstruction_loss = L.EuclideanLoss(n.reconstruction1, n.target_features, name="reconstruction_loss",
				loss_weight=l2_loss_weight, loss_param=dict(normalize=True)) 

	# now finish the computation of the rest of the network
	# hard coded for measuring equivariance of last hidden layers
	n.classify = L.InnerProduct(n.reconstruction2, num_output=num_classes, name="classify",
		param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult': 0, 'decay_mult': 0}])  # weights to be fixed to the network's original weights
	n.prob = L.Softmax(n.classify, name='prob')
	n.ce_loss = L.EuclideanLoss(n.prob, n.target_output_probs, name='ce_loss',
		loss_weight=0, loss_param=dict(normalize=True))
	n.ce_loss2 = L.SoftmaxWithLoss(n.classify, n.labels, name='ce_loss2',
		loss_weight=ce_loss_weight, loss_param=dict(normalize=True))

	# TODO: write Full Softmax/CE loss layer to compare full distributions
	#n.ce_loss = L.SoftmaxCrossEntropyLossLayer(n.prev, n.transform_output_probs)

	n.accuracy = L.Accuracy(n.classify, n.labels)

	return n.to_proto()


def create_solver(args, num_train_instances, num_test_instances):
	s = caffe.proto.caffe_pb2.SolverParameter()
	s.net = args.train_file

	s.test_interval = num_train_instances / args.train_batch_size / 4
	s.test_iter.append(num_test_instances)
	s.max_iter = num_train_instances / args.train_batch_size * args.max_epochs

	#s.solver_type = caffe.proto.caffe_pb2.SolverType.SGD  # why isn't it working?  Default anyway
	s.momentum = 0.9
	s.weight_decay = 1e-3

	s.base_lr = args.learning_rate
	s.monitor_test = True
	s.monitor_test_id = 0
	s.test_compute_loss = True
	s.max_steps_without_improvement = 4
	s.max_periods_without_improvement = 1
	s.gamma = 0.1

	s.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU
	s.snapshot = 0  # don't do snapshotting
	s.display = 100

	return s
	

def init_model(network_file, weights_file, gpu=0):
	if args.gpu >= 0:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()

	model = caffe.Net(network_file, weights_file, caffe.TEST)
	return model


def open_lmdb(test_lmdb):
	env = lmdb.open(test_lmdb, readonly=True, map_size=int(2 ** 42))
	txn = env.begin(write=False)
	cursor = txn.cursor()
	cursor.first()
	return env, txn, cursor


def log(args, s, newline=True):
	print s
	if args.log_file:
		if not hasattr(args, 'log'):
			args.log = open(args.log_file, 'w')
		if newline:
			args.log.write("%s\n" % s)
		else:
			args.log.write(s)


def apply_elastic_deformation(im, tokens):
	sigma, alpha, seed = float(tokens[1]), float(tokens[2]), int(tokens[3])
	np.random.seed(seed)

	displacement_x = np.random.uniform(-1 * alpha, alpha, im.shape[:2])
	displacement_y = np.random.uniform(-1 * alpha, alpha, im.shape[:2])

	displacement_x = scipy.ndimage.gaussian_filter(displacement_x, sigma, truncate=2)
	displacement_y = scipy.ndimage.gaussian_filter(displacement_y, sigma, truncate=2)

	coords_y = np.asarray( [ [y] * im.shape[1] for y in xrange(im.shape[0]) ])
	coords_y = np.clip(coords_y + displacement_y, 0, im.shape[0])

	coords_x = np.transpose(np.asarray( [ [x] * im.shape[0] for x in xrange(im.shape[1]) ] ))
	coords_x = np.clip(coords_x + displacement_x, 0, im.shape[1])

	# the backwards mapping function, which assures that all coords are in
	# the range of the input
	if im.ndim == 3:
		coords_y = coords_y[:,:,np.newaxis]
		coords_y = np.concatenate(im.shape[2] * [coords_y], axis=2)[np.newaxis,:,:,:]

		coords_x = coords_x[:,:,np.newaxis]
		coords_x = np.concatenate(im.shape[2] * [coords_x], axis=2)[np.newaxis,:,:,:]

		coords_d = np.zeros_like(coords_x)
		for x in xrange(im.shape[2]):
			coords_d[:,:,:,x] = x
		coords = np.concatenate( (coords_y, coords_x, coords_d), axis=0)
	else:
		coords = np.concatenate( (coords_y[np.newaxis,:,:], coords_x[np.newaxis,:,:]), axis=0)

	## first order spline interpoloation (bilinear?) using the backwards mapping
	output = scipy.ndimage.map_coordinates(im, coords, order=1, mode='reflect')

	return output



# "crop y x height width"
def apply_crop(im, tokens):
	y, x = int(tokens[1]), int(tokens[2])
	height, width = int(tokens[3]), int(tokens[4])
	if y >= im.shape[0] or x >= im.shape[1]:
		print "Invalid crop: (y,x) outside image bounds (%r with %r)" % (im.shape, tokens)
		exit(1)
	if (y < 0 and y + height >= 0) or (x < 0 and x + width >= 0):
		print "Invalid crop: negative indexing has wrap around (%r with %r)" % (im.shape, tokens)
		exit(1)
	if (height > im.shape[0]) or (width > im.shape[1]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, tokens)
		exit(1)
	if (y + height > im.shape[0]) or (x + width > im.shape[1]):
		print "Invalid crop: crop goes off edge of image (%r with %r)" % (im.shape, tokens)
		exit(1)
		
	return im[y:y+height,x:x+width]


def apply_rand_crop(im, tokens):
	height, width = int(tokens[1]), int(tokens[2])
	if (height > im.shape[0]) or (width > im.shape[1]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, tokens)
		exit(1)

	y = random.randint(0, im.shape[0] - height)
	x = random.randint(0, im.shape[1] - width)
	return im[y:y+height,x:x+width]

# "resize height width"
def apply_resize(im, tokens):
	size = int(tokens[2]), int(tokens[1])
	return cv2.resize(im, size)

# "mirror {h,v,hv}"
def apply_mirror(im, tokens):
	if tokens[1] == 'h':
		return cv2.flip(im, 0)
	elif tokens[1] == 'v':
		return cv2.flip(im, 1)
	elif tokens[1] == 'hv':
		return cv2.flip(im, -1)
	else:
		print "Unrecongized mirror operation %r" % tokens
		exit(1)

def apply_color_jitter(im, tokens):
	sigma, seed = float(tokens[1]), int(tokens[2])
	np.random.seed(seed)
	im = im.astype(int)  # protect against over-flow wrapping
	if im.shape == 2:
		im = im + int(np.random.normal(0, sigma))
	else:
		for c in xrange(im.shape[2]):
			im[:,:,c] = im[:,:,c] + int(np.random.normal(0, sigma))
	
	# truncate back to image range
	im = np.clip(im, 0, 255)
	im = im.astype(np.uint8) 
	return im

# "guassnoise sigma seed"
def apply_gaussnoise(im, tokens):
	sigma, seed = float(tokens[1]), int(tokens[2])
	np.random.seed(seed)
	noise = np.random.normal(0, sigma, im.shape[:2])
	if len(im.shape) == 2:
		im = (im + noise)
	else:
		im = im + noise[:,:,np.newaxis]
	im = np.clip(im, 0, 255)
	im = im.astype(np.uint8)
	return im

# "rotate degree"
def apply_rotation(im, tokens):
	degree = float(tokens[1])
	center = (im.shape[0] / 2, im.shape[1] / 2)
	rot_mat = cv2.getRotationMatrix2D(center, degree, 1.0)
	return cv2.warpAffine(im, rot_mat, im.shape[:2], flags=cv2.INTER_LINEAR)

# "blur sigma"
def apply_blur(im, tokens):
	sigma = float(tokens[1])
	size = int(sigma * 4 + .999)
	if size % 2 == 0:
		size += 1
	return cv2.GaussianBlur(im, (size, size), sigma)
	
# "unsharpmask sigma amount"
def apply_unsharpmask(im, tokens):
	blurred = np.atleast_3d(apply_blur(im, tokens))
	amount = float(tokens[2])
	sharp = (1 + amount) * im + (-amount * blurred)
	sharp = np.clip(sharp, 0, 255)
	return sharp

# "shear degree {h,v}"
def apply_shear(im, tokens):
	degree = float(tokens[1])
	radians = math.tan(degree * math.pi / 180)
	shear_mat = np.array([ [1, 0, 0], [0, 1, 0] ], dtype=np.float)
	if tokens[2] == 'h':
		shear_mat[0,1] = radians
	elif tokens[2] == 'v':
		shear_mat[1,0] = radians
	else:
		print "Invalid shear type: %r" % tokens
	return cv2.warpAffine(im, shear_mat, im.shape[:2], flags=cv2.INTER_LINEAR)

# "perspective dy1 dx1 dy2 dx2 dy3 dx3 dy4 dx4"
def apply_perspective(im, tokens):
	pts1 = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
	pts2 = np.array([[0 + float(tokens[1]) ,0 + float(tokens[2])],
					   [1 + float(tokens[3]) ,0 + float(tokens[4])],
					   [1 + float(tokens[5]) ,1 + float(tokens[6])],
					   [0 + float(tokens[7]) ,1 + float(tokens[8])]
					   ], dtype=np.float32)
	M = cv2.getPerspectiveTransform(pts1,pts2)
	return cv2.warpPerspective(im, M, im.shape[:2])

def apply_transform(im, transform_str):
	tokens = transform_str.split()
	if tokens[0] == 'crop':
		return apply_crop(im, tokens)
	if tokens[0] == 'randcrop':
		return apply_rand_crop(im, tokens)
	elif tokens[0] == 'resize':
		return apply_resize(im, tokens)
	elif tokens[0] == 'mirror':
		return apply_mirror(im, tokens)
	elif tokens[0] == 'gaussnoise':
		return apply_gaussnoise(im, tokens)
	elif tokens[0] == 'rotation':
		return apply_rotation(im, tokens)
	elif tokens[0] == 'blur':
		return apply_blur(im, tokens)
	elif tokens[0] == 'unsharpmask' or tokens[0] == 'unsharp':
		return apply_unsharpmask(im, tokens)
	elif tokens[0] == 'shear':
		return apply_shear(im, tokens)
	elif tokens[0] == 'perspective':
		return apply_perspective(im, tokens)
	elif tokens[0] == 'color_jitter':
		return apply_color_jitter(im, tokens)
	elif tokens[0] == 'elastic':
		return apply_elastic_deformation(im, tokens)
	elif tokens[0] == 'none':
		return im
	else:
		print "Unknown transform: %r" % transform_str
		exit(1)


# all transforms must yield images of the same dimensions
def apply_transforms(im, multi_transform_str):
	transform_strs = multi_transform_str.split(';')
	for ts in transform_strs:
		im = apply_transform(im, ts)
	return im


def apply_all_transforms(im, transform_strs):
	ims = list()
	for ts in transform_strs:
		im_out = apply_transforms(im, ts)
		ims.append(im_out)
	return ims


def get_transforms(transform_file):
	transforms = list()
	if transform_file:
		transforms = map(lambda s: s.rstrip().lower(), open(args.transform_file, 'r').readlines())
	transforms = filter(lambda s: not s.startswith("#"), transforms)
	if 'none' not in transforms:
		transforms.append('none')
	return transforms


def get_image(cursor, args):
	doc_datum = caffe.proto.caffe_pb2.DocumentDatum()
	dd_serialized = cursor.value()
	if not dd_serialized:
		# done traversing database
		return None, None
	cursor.next()
	doc_datum.ParseFromString(dd_serialized)  # decode lmdb entry
	label = doc_datum.dbid

	nparr = np.fromstring(doc_datum.image.data, np.uint8)
	# load the image as is (do not force color/grayscale)
	im = cv2.imdecode(nparr, args.color)  
	if im.ndim == 2:
		# explicit single channel to match dimensions of color
		im = im[:,:,np.newaxis]

	return im, label


def preprocess_im(im, means, scale):
	means = np.asarray(map(float, means.split(',')))
	means = means[np.newaxis,np.newaxis,np.newaxis,:]
	return float(scale) * (im - means)


def fprop(model, ims, args):
	# batch up all transforms at once
	transposed = np.transpose(ims, [0,3,1,2])
	model.blobs[args.input_blob].reshape(*transposed.shape)
	model.blobs[args.input_blob].data[:,:,:,:] = transposed
	#for x in xrange(ims.shape[0]):
	#	transposed = np.transpose(ims[0], [0,3,1,2])
	#	model.blobs[args.input_blob].data[x,:,:,:] = transposed
	# propagate on batch
	model.forward()

	
def scale_shift_im(im, means, scale):
	preprocessed_im = scale * (im - means)
	return preprocessed_im


def get_activations(model, transforms, lmdb_file, args):
	env, txn, cursor = open_lmdb(lmdb_file)

	num_images = min(args.max_images, env.stat()['entries'])
	num_activations = model.blobs[args.blob].data.shape[1]
	num_classes = model.blobs['prob'].data.shape[1]

	activations = {transform: np.zeros((num_images, num_activations)) for transform in transforms}
	output_probs = {transform: np.zeros((num_images, num_classes))  for transform in transforms}
	classifications = {transform: np.zeros((num_images,))  for transform in transforms}
	labels = list()
	
	means = np.asarray(map(int, args.means.split(',')))
	scale = args.scale

	for iter_num in xrange(num_images):
		im, label = get_image(cursor, args)
		labels.append(label)
		ims = apply_all_transforms(im, transforms)
		ims = map(lambda im: scale_shift_im(im, means, scale), ims)

		fprop(model, ims, args)
		for idx, transform in enumerate(transforms):
			activations[transform][iter_num, :] = model.blobs[args.blob].data[idx, :]
			output_probs[transform][iter_num, :] = model.blobs['prob'].data[idx, :]
			classifications[transform][iter_num] = np.argmax(model.blobs['prob'].data[idx, :])

		if iter_num > 0 and iter_num % 100 == 0:
			log(args, "%.2f%% (%d/%d) Batches" % (100. * iter_num / num_images, iter_num, num_images))
		log(args, "%.2f%% (%d/%d) Batches" % (100. * iter_num / num_images, iter_num, num_images))

	return activations, output_probs, classifications, np.asarray(labels)


def measure_avg_l2(a, b):
	total_euclidean_dist = 0
	for idx in xrange(a.shape[0]):
		total_euclidean_dist += np.sqrt(np.sum((a[idx] - b[idx]) ** 2))
	return total_euclidean_dist / a.shape[0]
	

def measure_avg_jsd(a, b):
	total_divergence = 0
	for idx in xrange(a.shape[0]):
		m = 0.5 * (a[idx] + b[idx])
		jsd = 0.5 * (scipy.stats.entropy(a[idx], m) + scipy.stats.entropy(b[idx], m))
		total_divergence += math.sqrt(jsd)
	return total_divergence / a.shape[0]


def measure_agreement(a, b):
	return np.sum(a == b) / float(a.shape[0])


def measure_invariances(all_features, all_output_probs, all_classifications, labels, transforms, args):
	# Avg L2 distance between features
	# % agreement on predicted labels
	# Avg sqrt(Jensen Shannon Divergence) between output_probs
	# All of the above for instances correctly classified
	metrics = {transform: dict() for transform in transforms}
	original_features = all_features['none']
	original_output_probs = all_output_probs['none']
	original_classifications = all_classifications['none']

	correct_indices = (original_classifications == labels)
	for transform in transforms:
		transform_features = all_features[transform]
		transform_output_probs = all_output_probs[transform]
		transform_classifications = all_classifications[transform]

		avg_l2 = measure_avg_l2(transform_features, original_features)
		metrics[transform]['avg_l2'] = avg_l2
		c_avg_l2 = measure_avg_l2(transform_features[correct_indices], original_features[correct_indices])
		metrics[transform]['c_avg_l2'] = c_avg_l2

		agreement = measure_agreement(transform_classifications, original_classifications)
		metrics[transform]['agreement'] = agreement
		c_agreement = measure_agreement(transform_classifications[correct_indices], original_classifications[correct_indices])
		metrics[transform]['c_agreement'] = c_agreement

		accuracy = measure_agreement(transform_classifications, labels)
		metrics[transform]['accuracy'] = accuracy
		c_accuracy = measure_agreement(transform_classifications[correct_indices], labels[correct_indices])
		metrics[transform]['c_accuracy'] = c_accuracy

		avg_jsd = measure_avg_jsd(transform_output_probs, original_output_probs)
		metrics[transform]['avg_jsd'] = avg_jsd
		c_avg_jsd = measure_avg_jsd(transform_output_probs[correct_indices], original_output_probs[correct_indices])
		metrics[transform]['c_avg_jsd'] = c_avg_jsd

	return metrics


def write_hdf5s(args, input_train_features, target_train_features, target_train_output_probs, train_labels,
	input_test_features, target_test_features, target_test_output_probs, test_labels):

	with h5py.File(args.train_db, 'w') as f:
		f['input_features'] = input_train_features.astype(np.float32)
		f['target_features'] = target_train_features.astype(np.float32)
		f['target_output_probs'] = target_train_output_probs.astype(np.float32)
		f['labels'] = train_labels[:,np.newaxis].astype(np.float32)

	with h5py.File(args.test_db, 'w') as f:
		f['input_features'] = input_test_features.astype(np.float32)
		f['target_features'] = target_test_features.astype(np.float32)
		f['target_output_probs'] = target_test_output_probs.astype(np.float32)
		f['labels'] = test_labels[:,np.newaxis].astype(np.float32)


def train_equivariance_model(model_type, loss, input_train_features, input_test_features, target_train_features, 
	target_test_features, train_labels, test_labels, target_train_output_probs, target_test_output_probs, 
	classifier_weights, classifier_bias, args):

	num_train_instances = input_train_features.shape[0]
	num_test_instances = input_test_features.shape[0]
	num_activations = input_train_features.shape[1]
	num_classes = target_train_output_probs.shape[1]
	l2_loss_weight = 1 if loss == 'l2' else 0
	mlp = model_type == 'mlp'

	write_hdf5s(args, input_train_features, target_train_features, target_train_output_probs, train_labels,
			input_test_features, target_test_features, target_test_output_probs, test_labels)

	# Write the prototxt for the train and test nets, as well as the solver
	net_param = equivariance_proto(args, num_activations, num_classes, l2_loss_weight, mlp)
	with open(args.train_file, 'w') as f:
		f.write(re.sub("VAL_", "", str(net_param)))

	solver_param = create_solver(args, num_train_instances, num_test_instances)
	with open(args.solver_file, 'w') as f:
		f.write(str(solver_param))

	# load the solver and the network files it references
	solver = caffe.get_solver(args.solver_file)

	# fix the classificaiton weights/biases to be the passed in weights/biases
	classify_layer_params = solver.net.params['classify']
	classify_layer_params[0].data[:] = classifier_weights[:]  # data copy, not object reassignment
	classify_layer_params[1].data[:] = classifier_bias[:]

	#classify_layer_params = solver.test_nets[0].params['classify']
	#classify_layer_params[0].data[:] = classifier_weights[:] 
	#classify_layer_params[1].data[:] = classifier_bias[:]

	# initialize mappings to the identity
	mapping_layer_params = solver.net.params['mapping']
	shape = mapping_layer_params[0].data.shape
	mapping_layer_params[0].data[:] = np.eye(shape[0], shape[1])[:]

	#mapping_layer_params = solver.test_nets[0].params['mapping']
	#shape = mapping_layer_params[0].data.shape
	#mapping_layer_params[0].data[:] = np.eye(shape[0], shape[1])[:]

	if mlp:
		hidden_layer_params = solver.net.params['mlp_hidden']
		shape = hidden_layer_params[0].data.shape
		hidden_layer_params[0].data[:] = np.eye(shape[0], shape[1])[:]

		#hidden_layer_params = solver.test_nets[0].params['mlp_hidden']
		#shape = hidden_layer_params[0].data.shape
		#hidden_layer_params[0].data[:] = np.eye(shape[0], shape[1])[:]
		

	#solver.net.forward()
	#solver.test_nets[0].forward()
	solver.solve()

	return solver.net, solver.test_nets[0], solver.iter  # the trained model


def measure_equivariances(train_features, all_train_labels, train_classifications, train_output_probs,
		test_features, all_test_labels, test_classifications, test_output_probs, transforms, model, args):
	all_train_metrics = {transform: dict() for transform in transforms}
	all_test_metrics = {transform: dict() for transform in transforms}
	# model_loss_metric
	#all_train_metrics['none'] = dict()
	#for data in ['', 'c_']:
	#	for model_type in ['linear', 'mlp']:
	#		for loss in ['l2', 'ce']:
	#			for metric in ['acc', 'agreement', 'l2', 'jsd']:
	#				for mode in ['train', 'test']:
	#					all_train_metrics['none']["%s_%s%s_%s_%s" % (mode, data, model_type, loss, metric)] = 0

	# Get indices for different data splits (all vs those classified correctly under no transform)
	original_train_classifications = train_classifications['none']
	original_test_classifications = test_classifications['none']
	train_correct_indices = original_train_classifications == all_train_labels
	test_correct_indices = original_test_classifications == all_test_labels
	train_all_indices = np.ones_like(train_correct_indices, dtype=bool)
	test_all_indices = np.ones_like(test_correct_indices, dtype=bool)

	last_layer_params = model.params.items()[-1][1]
	classification_weights = last_layer_params[0].data
	classification_bias = last_layer_params[1].data

	for transform in transforms:
		if transform == 'none':
			continue
		#for data in ['', 'c_']:
		for data in ['']:
			if data == 'c_':
				train_indices = train_correct_indices
				test_indices = test_correct_indices
			else:
				train_indices = train_all_indices
				test_indices = test_all_indices
			transform_train_features = train_features[transform][train_indices]
			transform_test_features = test_features[transform][test_indices]
			transform_train_output_probs = train_output_probs[transform][train_indices]
			transform_test_output_probs = test_output_probs[transform][test_indices]
			transform_train_classifications = train_classifications[transform][train_indices]
			transform_test_classifications = test_classifications[transform][test_indices]

			train_labels = all_train_labels[train_indices]
			test_labels = all_test_labels[train_indices]
			original_train_features = train_features['none'][train_indices]
			original_test_features = test_features['none'][test_indices]
			original_train_output_probs = train_output_probs['none'][train_indices]
			original_test_output_probs = test_output_probs['none'][test_indices]
			original_train_classifications = train_classifications['none'][train_indices]
			original_test_classifications = test_classifications['none'][test_indices]


			# l2
			# predict the transformed representation directly from the original representation
			for model_type in ['linear', 'mlp']:
				train_metrics, test_metrics = _measure_equivariance('lr', 'l2', original_train_features, original_test_features,
					transform_train_features, transform_test_features, train_labels, test_labels, transform_train_output_probs,
					transform_test_output_probs, classification_weights, classification_bias, transform_train_classifications,
					transform_test_classifications, args)
				for metric, val in train_metrics.iteritems():
					all_train_metrics[transform]["%s%s_%s_%s" % (data, model_type, 'l2', metric)] = val
				for metric, val in test_metrics.iteritems():
					all_test_metrics[transform]["%s%s_%s_%s" % (data, model_type, 'l2', metric)] = val

			# CE loss
			# the classification weights which are fixed are trained for the original representation, so we take the transformed
			# represenation and try to undo it using a linear or mlp model
			for model_type in ['linear', 'mlp']:
				train_metrics, test_metrics = _measure_equivariance('lr', 'ce', transform_train_features, transform_test_features,
					original_train_features, original_test_features, train_labels, test_labels, original_train_output_probs,
					original_test_output_probs, classification_weights, classification_bias, original_train_classifications, 
					original_test_classifications, args)
				for metric, val in train_metrics.iteritems():
					all_train_metrics[transform]["%s%s_%s_%s" % (data, model_type, 'ce', metric)] = val
				for metric, val in test_metrics.iteritems():
					all_test_metrics[transform]["%s%s_%s_%s" % (data, model_type, 'ce', metric)] = val

	return all_train_metrics, all_test_metrics

def _measure_equivariance(model_type, loss, input_train_features, input_test_features, target_train_features, 
			target_test_features, train_labels, test_labels, target_train_output_probs, target_test_output_probs, 
			classification_weights, classification_bias, target_train_classifications, target_test_classifications, args):
	num_train_instances = input_train_features.shape[0]
	num_test_instances = input_test_features.shape[0]
	num_activations = input_test_features.shape[1]
	num_classes = target_test_output_probs.shape[1]

	train_model, test_model, num_iters = train_equivariance_model(model_type, loss, input_train_features, input_test_features, 
			target_train_features, target_test_features, train_labels, test_labels, target_train_output_probs,
			target_test_output_probs, classification_weights, classification_bias, args)

	# misaligned since it pulls data from where training left off
	train_metrics = score_model(train_model, target_train_features, target_train_output_probs, 
		target_train_classifications, train_labels, num_train_instances, num_iters)

	# should be aligned because testing the model during training should always completely
	# cycle through the db
	test_metrics = score_model(test_model, target_test_features, target_test_output_probs, 
		target_test_classifications, test_labels, num_test_instances, 0)
					
	return train_metrics, test_metrics


def extract_from_equivariant(model, num_instances, offset):
	num_activations = model.blobs['reconstruction'].data.shape[1]
	num_classes = model.blobs['prob'].data.shape[1]

	features = np.zeros((num_instances, num_activations))
	output_probs = np.zeros((num_instances, num_classes))
	classifications = np.zeros((num_instances,))

	for idx in xrange(num_instances):
		arr_idx = (idx + offset) % num_instances
		model.forward()
		features[arr_idx,:] = model.blobs['reconstruction'].data[0,:]
		output_probs[arr_idx,:] = model.blobs['prob'].data[0,:]
		classifications[arr_idx] = np.argmax(model.blobs['prob'].data[0])

	return features, output_probs, classifications
	


def score_model(model, target_features, target_output_probs, 
	target_classifications, target_labels, num_instances, offset):
	reconstructed_features, predicted_output_probs, predicted_classifications = extract_from_equivariant(model, num_instances, offset)
	metrics = dict()

	avg_l2 = measure_avg_l2(reconstructed_features, target_features)
	metrics['avg_l2'] = avg_l2

	agreement = measure_agreement(predicted_classifications, target_classifications)
	metrics['agreement'] = agreement

	accuracy = measure_agreement(predicted_classifications, target_labels)
	metrics['accuracy'] = accuracy

	avg_jsd = measure_avg_jsd(predicted_output_probs, target_output_probs)
	metrics['avg_jsd'] = avg_jsd

	return metrics


def main(args):
	log(args, str(args))
	transforms = get_transforms(args.transform_file)
	log(args, "Loaded Transforms.  %d transforms" % len(transforms))
	model = init_model(args.network_file, args.weight_file, gpu=args.gpu)

	train_features, train_output_probs, train_classifications, train_labels = get_activations(model, transforms, args.train_lmdb, args)
	test_features, test_output_probs, test_classifications, test_labels = get_activations(model, transforms, args.test_lmdb, args)

	log(args, "Measuring invariances...")
	train_invariance_metrics = measure_invariances(train_features, train_output_probs, train_classifications, train_labels, transforms, args)
	test_invariance_metrics = measure_invariances(test_features, test_output_probs, test_classifications, test_labels, transforms, args)
	log(args, "Done...")

	setup_scratch_space(args)
	log(args, "Measuring equivariances...")
	equivariance_metrics = measure_equivariances(train_features, train_labels, train_classifications, train_output_probs, 
			test_features, test_labels, test_classifications, test_output_probs, transforms, model, args)

	log(args, "Invariance Train Metrics:\n %s" % pprint.pformat(train_invariance_metrics))
	log(args, "Invariance Test Metrics:\n %s" % pprint.pformat(test_invariance_metrics))
	log(args, "Equivariance Train Metrics:\n %s" % pprint.pformat(equivariance_metrics[0]))
	log(args, "Equivariance Test Metrics:\n %s" % pprint.pformat(equivariance_metrics[1]))
	log(args, "Done...")

	if args.log_file:
		args.log.close()

	#cleanup_scratch_space(args)
		

def get_args():
	parser = argparse.ArgumentParser(
		description="Measures invariance of learned representations with respect to the given transforms")
	parser.add_argument("network_file", 
				help="Caffe network file")
	parser.add_argument("weight_file", 
				help="Caffe weights file")
	parser.add_argument("train_lmdb", 
				help="LMDB of images (encoded DocDatums), used to train equivariance mappings")
	parser.add_argument("test_lmdb", 
				help="LMDB of images (encoded DocDatums), used to test equivariance mappings")
	parser.add_argument("transform_file", type=str, default="",
				help="File containing transformations to do")

	parser.add_argument("-f", "--log-file", type=str, default="",
				help="Log File")
	parser.add_argument("-m", "--means", type=str, default="0",
				help="Optional mean values per channel " 
				"(e.g. 127 for grayscale or 182,192,112 for BGR)")
	parser.add_argument("-a", "--scale", type=float, default=(1.0 / 255),
				help="Optional scale factor")
	#parser.add_argument("-b", "--train-batch-size", type=int, default=4,
	#			help="Batch size for training equivariance models")
	parser.add_argument("-i", "--test-iterations", type=int, default=400,
				help="Number of test iterations for testing equivariance models")
	parser.add_argument("-e", "--max-epochs", type=int, default=20,
				help="Max training epochs for equivariance models")
	parser.add_argument("-l", "--learning-rate", type=float, default=0.1,
				help="Initial Learning rate for equivariance models")
	parser.add_argument("-k", "--hidden-size", type=float, default=1.,
				help="Fraction of # inputs to determine hidden size for mlp equivariant mappings")
	parser.add_argument("--max-images", default=40000, type=int, 
				help="Max number of images for processing")
	parser.add_argument("--gpu", type=int, default=0,
				help="GPU to use for running the models")
	parser.add_argument("--color", default=False, action="store_true",
				help="Read in the images as color")
	parser.add_argument("--input-blob", type=str, default="data",
				help="Name of input blob")
	parser.add_argument("--blob", type=str, default="fc7",
				help="Name of blob on which to measure equivariance")

	args = parser.parse_args()
	args.train_batch_size = 1

	return args
	

if __name__ == "__main__":
	args = get_args()
	main(args)

