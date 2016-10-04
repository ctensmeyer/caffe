
import os
import re
import sys
import caffe  # must come before cv2
import cv2
import h5py
import math
import lmdb
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
from utils import get_transforms, apply_all_transforms, safe_mkdir


def setup_scratch_space(args):
	#safe_mkdir("./tmp")
	#args.tmp_dir = "./tmp"
	args.tmp_dir = tempfile.mkdtemp()
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


def equivariance_proto(args, num_features, num_classes, loss='l2', mlp=False):
	if loss == 'l2':
		ce_hard_loss_weight = 0
		ce_soft_loss_weight = 0
		l2_loss_weight = 1
	elif loss == 'ce_hard':
		ce_hard_loss_weight = 0.75
		ce_soft_loss_weight = 0
		l2_loss_weight = 0.25
	else:
		ce_hard_loss_weight = 0
		ce_soft_loss_weight = 0.75
		l2_loss_weight = 0.25

	n = caffe.NetSpec()

	n.input_features, n.target_features, n.target_output_probs, n.labels = L.HDF5Data(
		batch_size=args.train_batch_size, source=args.train_db_list, ntop=4, include=dict(phase=caffe.TRAIN))
	n.VAL_input_features, n.VAL_target_features, n.VAL_target_output_probs, n.VAL_labels = L.HDF5Data(
		batch_size=1, source=args.test_db_list, ntop=4, include=dict(phase=caffe.TEST))

	if mlp:
		n.prev = L.InnerProduct(n.input_features, num_output=int(args.hidden_size * num_features), name='mlp_hidden',
			weight_filler={'type': 'gaussian', 'std': 0.001,})
		n.prev = L.TanH(n.prev, in_place=True)

		n.residual = L.InnerProduct(n.prev, num_output=num_features, name='residual',
			weight_filler={'type': 'gaussian', 'std': 0.001,})
		n.combined = L.Eltwise(n.input_features, n.residual, name='combined')
		n.reconstruction = L.ReLU(n.combined, name='reconstruction')  # assumes that target_features are rectified
	else:
		n.residual = L.InnerProduct(n.input_features, num_output=num_features, name='residual',
			weight_filler={'type': 'gaussian', 'std': 0.001,})
		n.combined = L.Eltwise(n.input_features, n.residual, name='combined')
		n.combined = L.Eltwise(n.input_features, n.residual, name='combined')
		n.reconstruction = L.ReLU(n.combined, name='reconstruction')  # assumes that target_features are rectified

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

	n.prob = L.Softmax(n.classify)
	# use the original predicted probs as the targets for CE
	n.ce_loss_soft = L.SoftmaxFullLoss(n.classify, n.target_output_probs, name='ce_loss_soft',
		loss_weight=ce_soft_loss_weight, loss_param=dict(normalize=True))

	# use the labels as the targets for CE
	n.ce_loss_hard = L.SoftmaxWithLoss(n.classify, n.labels, name='ce_loss_hard',
		loss_weight=ce_hard_loss_weight, loss_param=dict(normalize=True))

	# use n.prob to suppress it as an output of the network
	n.accuracy = L.Accuracy(n.prob, n.labels)

	return n.to_proto()


def create_solver(args, num_train_instances, num_test_instances):
	s = caffe.proto.caffe_pb2.SolverParameter()
	s.net = args.train_file

	s.test_interval = num_train_instances / args.train_batch_size / 4
	s.test_iter.append(num_test_instances)
	s.max_iter = num_train_instances / args.train_batch_size * args.max_epochs

	#s.solver_type = caffe.proto.caffe_pb2.SolverType.SGD  # why isn't it working?  Default anyway
	s.momentum = 0.9
	s.weight_decay = 5e-3  # strong weight decay as a prior to the identity mapping

	s.base_lr = args.learning_rate
	s.monitor_test = True
	s.monitor_test_id = 0
	s.test_compute_loss = True
	s.max_steps_without_improvement = 4
	s.max_periods_without_improvement = 1
	s.gamma = 0.1

	s.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU
	s.snapshot = 0  # don't do snapshotting
	s.snapshot_after_train = False
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


def log(args, s, newline=True):
	print s
	if args.log_file:
		if not hasattr(args, 'log'):
			args.log = open(args.log_file, 'w')
		if newline:
			args.log.write("%s\n" % s)
		else:
			args.log.write(s)



def open_dbs(db_paths):
	dbs = list()
	for path in db_paths:
		env = lmdb.open(path, readonly=True, map_size=int(2 ** 42))
		txn = env.begin(write=False)
		cursor = txn.cursor()
		cursor.first()
		dbs.append( (env, txn, cursor) )
	return dbs


def close_dbs(dbs):
	for env, txn, cursor in dbs:
		txn.abort()
		env.close()


def fprop(model, ims, args):
	# batch up all transforms at once
	transposed = np.transpose(ims, [0,3,1,2])
	model.blobs[args.input_blob].reshape(*transposed.shape)
	model.blobs[args.input_blob].data[:,:,:,:] = transposed
	model.forward()

	
def get_image(dd_serialized, slice_idx, args):
	doc_datum = caffe.proto.caffe_pb2.DocumentDatum()
	doc_datum.ParseFromString(dd_serialized)	

	channel_tokens = args.channels.split(args.delimiter)
	channel_idx = min(slice_idx, len(channel_tokens)-1)
	num_channels = int(channel_tokens[channel_idx])

	nparr = np.fromstring(doc_datum.image.data, np.uint8)
	im = cv2.imdecode(nparr, int(num_channels == 3) )
	if im.ndim == 2:
		# explicit single channel to match dimensions of color
		im = im[:,:,np.newaxis]
	label = doc_datum.dbid
	return im, label


def scale_shift_im(im, slice_idx, args):

	# find the correct mean values.  
	means_tokens = args.means.split(args.delimiter)
	mean_idx = min(slice_idx, len(means_tokens) - 1)
	mean_vals = np.asarray(map(int, means_tokens[mean_idx].split(',')))

	# find the correct scale value
	scale_tokens = args.scales.split(args.delimiter)
	scale_idx = min(slice_idx, len(scale_tokens) - 1)
	scale_val = float(scale_tokens[scale_idx]) 

	preprocessed_im = scale_val * (im - mean_vals)
	return preprocessed_im


def prepare_images(dbs, transforms, args):
	ims_slice_transforms = list()
	labels = list()
	keys = list()

	# apply the transformations to every slice of the image independently
	for slice_idx, entry in enumerate(dbs):
		env, txn, cursor = entry

		im_slice, label_slice = get_image(cursor.value(), slice_idx, args)
		cursor.next()
		transformed_slices = apply_all_transforms(im_slice, transforms)
		for transform_idx in xrange(len(transformed_slices)):
			transformed_slices[transform_idx] = scale_shift_im(transformed_slices[transform_idx], slice_idx, args)

		ims_slice_transforms.append(transformed_slices)
		labels.append(label_slice)
		keys.append(cursor.key())

	# check that all keys match
	key = keys[0]
	for slice_idx, _key in enumerate(keys):
		if _key != key:
			log(args, "WARNING!, keys differ %s vs %s for slices %d and %d" % (key, _key, 0, slice_idx))

	
	# check that all labels match
	label = labels[0]
	for slice_idx, _label in enumerate(labels):
		if _label != label:
			log(args, "WARNING!, key %s has differing labels: %d vs %d for slices %d and %d" % (key, label, _label, 0, slice_idx))

	# stack each set of slices (along channels) into a single numpy array
	num_transforms = len(ims_slice_transforms[0])
	num_slices = len(ims_slice_transforms)
	ims = list()
	for transform_idx in xrange(num_transforms):
		im_slices = list()
		for slice_idx in xrange(num_slices):
			im_slice = ims_slice_transforms[slice_idx][transform_idx]
			im_slices.append(np.atleast_3d(im_slice))
		whole_im = np.concatenate(im_slices, axis=2) # append along channels
		ims.append(whole_im)

	return ims, label


def get_activations(model, transforms, lmdb_files, args):
	dbs = open_dbs(lmdb_files)

	num_images = min(args.max_images, dbs[0][0].stat()['entries'])
	num_activations = model.blobs[args.blob].data.shape[1]
	num_classes = model.blobs['prob'].data.shape[1]

	activations = {transform: np.zeros((num_images, num_activations)) for transform in transforms}
	output_probs = {transform: np.zeros((num_images, num_classes))  for transform in transforms}
	classifications = {transform: np.zeros((num_images,))  for transform in transforms}
	labels = list()
	
	for iter_num in xrange(num_images):
		ims, label = prepare_images(dbs, transforms, args)
		labels.append(label)

		fprop(model, ims, args)
		for idx, transform in enumerate(transforms):
			activations[transform][iter_num, :] = model.blobs[args.blob].data[idx, :]
			output_probs[transform][iter_num, :] = model.blobs['prob'].data[idx, :]
			classifications[transform][iter_num] = np.argmax(model.blobs['prob'].data[idx, :])

		if iter_num > 0 and iter_num % 10 == 0:
			log(args, "%.2f%% (%d/%d) Batches" % (100. * iter_num / num_images, iter_num, num_images))
	labels = np.asarray(labels)

	if args.shuffle:
		p = np.random.permutation(activations[transforms[0]].shape[0])
		labels = labels[p]
		for transform in transforms:
			activations[transform] = activations[transform][p]
			output_probs[transform] = output_probs[transform][p]
			classifications[transform] = classifications[transform][p]
	return activations, output_probs, classifications, labels


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
	original_features = all_features[transforms[0]]
	original_output_probs = all_output_probs[transforms[0]]
	original_classifications = all_classifications[transforms[0]]

	correct_indices = (original_classifications == labels)
	for transform in transforms:
		transform_features = all_features[transform]
		transform_output_probs = all_output_probs[transform]
		transform_classifications = all_classifications[transform]

		avg_l2 = measure_avg_l2(transform_features, original_features)
		metrics[transform]['avg_l2'] = avg_l2
		#c_avg_l2 = measure_avg_l2(transform_features[correct_indices], original_features[correct_indices])
		#metrics[transform]['c_avg_l2'] = c_avg_l2

		agreement = measure_agreement(transform_classifications, original_classifications)
		metrics[transform]['agreement'] = agreement
		#c_agreement = measure_agreement(transform_classifications[correct_indices], original_classifications[correct_indices])
		#metrics[transform]['c_agreement'] = c_agreement

		accuracy = measure_agreement(transform_classifications, labels)
		metrics[transform]['accuracy'] = accuracy
		#c_accuracy = measure_agreement(transform_classifications[correct_indices], labels[correct_indices])
		#metrics[transform]['c_accuracy'] = c_accuracy

		avg_jsd = measure_avg_jsd(transform_output_probs, original_output_probs)
		metrics[transform]['avg_jsd'] = avg_jsd
		#c_avg_jsd = measure_avg_jsd(transform_output_probs[correct_indices], original_output_probs[correct_indices])
		#metrics[transform]['c_avg_jsd'] = c_avg_jsd

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
	mlp = model_type == 'mlp'

	write_hdf5s(args, input_train_features, target_train_features, target_train_output_probs, train_labels,
			input_test_features, target_test_features, target_test_output_probs, test_labels)

	# Write the prototxt for the train and test nets, as well as the solver
	net_param = equivariance_proto(args, num_activations, num_classes, loss, mlp)
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

	solver.solve()
	return solver.net, solver.test_nets[0], solver.iter  # the trained model

def init_empty_metrics(transforms):
	d = dict()
	for transform in transforms:
		d[transform] = dict()
		for model_type in ['linear', 'mlp']:
			d[transform][model_type] = dict()
			for loss in ['l2', 'ce_soft', 'ce_hard']:
				d[transform][model_type][loss] = dict()
	return d


def measure_equivariances(train_features, all_train_labels, train_classifications, train_output_probs,
		test_features, all_test_labels, test_classifications, test_output_probs, transforms, model, args):
	all_train_metrics = init_empty_metrics(transforms) 
	all_test_metrics = init_empty_metrics(transforms)

	# Get indices for different data splits (all vs those classified correctly under no transform)
	original_train_classifications = train_classifications[transforms[0]]
	original_test_classifications = test_classifications[transforms[0]]
	train_correct_indices = original_train_classifications == all_train_labels
	test_correct_indices = original_test_classifications == all_test_labels
	train_all_indices = np.ones_like(train_correct_indices, dtype=bool)
	test_all_indices = np.ones_like(test_correct_indices, dtype=bool)

	last_layer_params = model.params.items()[-1][1]
	classification_weights = last_layer_params[0].data
	classification_bias = last_layer_params[1].data

	for transform in transforms:
		#if transform == transforms[0]:
		#	continue
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
			test_labels = all_test_labels[test_indices]
			original_train_features = train_features[transforms[0]][train_indices]
			original_test_features = test_features[transforms[0]][test_indices]
			original_train_output_probs = train_output_probs[transforms[0]][train_indices]
			original_test_output_probs = test_output_probs[transforms[0]][test_indices]
			original_train_classifications = train_classifications[transforms[0]][train_indices]
			original_test_classifications = test_classifications[transforms[0]][test_indices]

			# l2 training
			# predict the transformed representation directly from the original representation
			for model_type in ['linear', 'mlp']:
				train_metrics, test_metrics = _measure_equivariance(model_type, 'l2', original_train_features, original_test_features,
					transform_train_features, transform_test_features, train_labels, test_labels, transform_train_output_probs,
					transform_test_output_probs, classification_weights, classification_bias, transform_train_classifications,
					transform_test_classifications, args)
				for metric, val in train_metrics.iteritems():
					all_train_metrics[transform][model_type]['l2'][metric] = val
				for metric, val in test_metrics.iteritems():
					all_test_metrics[transform][model_type]['l2'][metric] = val

			# Cross Entropy training
			for suffix in ['_soft', '_hard']:
				loss = "ce%s" % suffix
				# the classification weights which are fixed are trained for the original representation, so we take the transformed
				# represenation and try to undo it using a linear or mlp model
				for model_type in ['linear', 'mlp']:
					train_metrics, test_metrics = _measure_equivariance(model_type, loss, transform_train_features, transform_test_features,
						original_train_features, original_test_features, train_labels, test_labels, original_train_output_probs,
						original_test_output_probs, classification_weights, classification_bias, original_train_classifications, 
						original_test_classifications, args)
					for metric, val in train_metrics.iteritems():
						all_train_metrics[transform][model_type][loss][metric] = val
					for metric, val in test_metrics.iteritems():
						all_test_metrics[transform][model_type][loss][metric] = val

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

def merge_all_dicts(dict_args):
	result = {}
	for dictionary in dict_args:
		result.update(dictionary)
	return result
		

def partition_transforms(transforms, size):
	base_transform = transforms[0]
	partitions = []
	idx = 1
	while idx < len(transforms):
		idx2 = 0
		partition = [base_transform]
		while idx2 < size and (idx + idx2) < len(transforms):
			partition.append(transforms[idx + idx2])
			idx2 += 1
		partitions.append(partition)
		idx += size
	return partitions


def filter_existing(transforms, out_dir):
	filtered_transforms = [transforms[0]]
	for transform in transforms:
		if not os.path.exists(os.path.join(out_dir, "%s.txt" % transform.replace(' ', '_'))):
			filtered_transforms.append(transform)
	
	return filtered_transforms

def write_output(out_dir, transform, train_invariance_metrics, test_invariance_metrics,
			train_equivariance_metrics, test_equivariance_metrics):
	all_metrics = {'train': 
					{'invariance': train_invariance_metrics,
					 'equivariance': train_equivariance_metrics},
				   'test': 
					{'invariance': test_invariance_metrics,
					 'equivariance': test_equivariance_metrics},
				}

	out_file = os.path.join(out_dir, "%s.txt" % transform.replace(' ', '_'))
	with open(out_file, 'w') as f:
		f.write(pprint.pformat(all_metrics))
	log(args, "Metrics for Transform %r:\n %s" % (transform, pprint.pformat(train_invariance_metrics)))


def main(args):
	log(args, str(args))

	safe_mkdir(args.out_dir)
	all_transforms, _ = get_transforms(args.transform_file)

	# don't redo work that we have already done
	all_transforms = filter_existing(all_transforms, args.out_dir)
	if len(all_transforms) <= 1:
		log(args, "No transforms to do.  Exiting...")
		exit()

	log(args, "Loaded Transforms.  %d transforms" % len(all_transforms))
	model = init_model(args.network_file, args.weight_file, gpu=args.gpu)

	train_lmdbs = args.train_lmdbs.split(args.delimiter)
	test_lmdbs = args.test_lmdbs.split(args.delimiter)

	base_transform = all_transforms[0]
	log(args, "Starting on Baseline Transform: %r\n" % base_transform)

	base_train_features, base_train_output_probs, base_train_classifications, _ = get_activations(model, [base_transform], train_lmdbs, args)
	base_test_features, base_test_output_probs, base_test_classifications, _ = get_activations(model, [base_transform], test_lmdbs, args)

	transform_partitions = partition_transforms(all_transforms, args.num_transforms)
	log(args, "Transform Partitions: %r" % transform_partitions)
	for transforms in transform_partitions:
		log(args, "Starting on Transforms: %r\n" % transforms)

		train_features, train_output_probs, train_classifications, train_labels = get_activations(model, transforms[1:], train_lmdbs, args)
		train_features.update(base_train_features)
		train_output_probs.update(base_train_output_probs)
		train_classifications.update(base_train_classifications)

		test_features, test_output_probs, test_classifications, test_labels = get_activations(model, transforms[1:], test_lmdbs, args)
		test_features.update(base_test_features)
		test_output_probs.update(base_test_output_probs)
		test_classifications.update(base_test_classifications)

		log(args, "Measuring invariances...")
		train_invariance_metrics = measure_invariances(train_features, train_output_probs, train_classifications, train_labels, transforms, args)
		test_invariance_metrics = measure_invariances(test_features, test_output_probs, test_classifications, test_labels, transforms, args)
		log(args, "Done...")

		setup_scratch_space(args)
		log(args, "Measuring equivariances...")
		train_equivariance_metrics, test_equivariance_metrics = measure_equivariances(train_features, train_labels, train_classifications, train_output_probs, 
				test_features, test_labels, test_classifications, test_output_probs, transforms, model, args)

		for transform in transforms:
			write_output(args.out_dir, transform, train_invariance_metrics[transform], test_invariance_metrics[transform],
					train_equivariance_metrics[transform], test_equivariance_metrics[transform])

	log(args, "Done Measure Equivariances")


	cleanup_scratch_space(args)
	log(args, "Exiting...")
		
	if args.log_file:
		args.log.close()

def check_args(args):
	num_train_lmdbs = 0 if args.train_lmdbs == "" else len(args.train_lmdbs.split(args.delimiter))
	num_test_lmdbs = 0 if args.test_lmdbs == "" else len(args.test_lmdbs.split(args.delimiter))
	if num_test_lmdbs == 0:
		raise Exception("No test lmdbs specified");
	if num_train_lmdbs != num_test_lmdbs:
		raise Exception("Different number of train and test lmdbs: %d vs %d" % (num_train_lmdb, num_test_lmdbs))

	num_scales = len(args.scales.split(args.delimiter))
	if num_scales != 1 and num_scales != num_test_lmdbs:
		raise Exception("Different number of test lmdbs and scales: %d vs %d" % (num_test_lmdbs, num_scales))

	num_means = len(args.means.split(args.delimiter))
	if num_means != 1 and num_means != num_test_lmdbs:
		raise Exception("Different number of test lmdbs and means: %d vs %d" % (num_test_lmdbs, num_means))

	num_channels = len(args.channels.split(args.delimiter))
	if num_channels != 1 and num_channels != num_test_lmdbs:
		raise Exception("Different number of test lmdbs and channels: %d vs %d" % (num_test_lmdbs, num_channels))

def get_args():
	parser = argparse.ArgumentParser(
		description="Measures invariance of learned representations with respect to the given transforms")
	parser.add_argument("network_file", 
				help="Caffe network file")
	parser.add_argument("weight_file", 
				help="Caffe weights file")
	parser.add_argument("train_lmdbs", 
				help="LMDB of images (encoded DocDatums), used to train equivariance mappings")
	parser.add_argument("test_lmdbs", 
				help="LMDB of images (encoded DocDatums), used to test equivariance mappings")
	parser.add_argument("transform_file", type=str,
				help="File containing transformations to do")
	parser.add_argument("out_dir", type=str,
				help="File to write the output")

	parser.add_argument("-f", "--log-file", type=str, default="",
				help="Log File")

	parser.add_argument("-m", "--means", type=str, default="0",
				help="Optional mean values per channel " 
				"(e.g. 127 for grayscale or 182,192,112 for BGR)")
	parser.add_argument('-c', '--channels', default="1", type=str,
				help='Number of channels to take from each slice')
	parser.add_argument("-a", "--scales", type=str, default=str(1.0 / 255),
				help="Optional scale factor")
	parser.add_argument("-d", "--delimiter", default=':', type=str, 
				help="Delimiter used for indicating multiple image slice parameters")
	parser.add_argument("--no-cache", default=False, action='store_true',
				help="Delimiter used for indicating multiple image slice parameters")

	parser.add_argument("-e", "--max-epochs", type=int, default=20,
				help="Max training epochs for equivariance models")
	parser.add_argument("-n", "--num-transforms", type=int, default=5,
				help="Max training epochs for equivariance models")
	parser.add_argument("-l", "--learning-rate", type=float, default=0.1,
				help="Initial Learning rate for equivariance models")
	parser.add_argument("-k", "--hidden-size", type=float, default=1.,
				help="Fraction of # inputs to determine hidden size for mlp equivariant mappings")
	parser.add_argument("--max-images", default=40000, type=int, 
				help="Max number of images for processing")
	parser.add_argument("--gpu", type=int, default=0,
				help="GPU to use for running the models")
	parser.add_argument("--input-blob", type=str, default="data",
				help="Name of input blob")
	parser.add_argument("--blob", type=str, default="fc7",
				help="Name of blob on which to measure equivariance")
	parser.add_argument("--shuffle", default=False, action="store_true",
				help="Name of blob on which to measure equivariance")

	args = parser.parse_args()
	check_args(args)
	args.train_batch_size = 1

	return args
	

if __name__ == "__main__":
	args = get_args()
	main(args)

