
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
	safe_mkdir("./tmp")
	args.tmp_dir = "./tmp"
	#args.tmp_dir = tempfile.mkdtemp()
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

		n.prev = L.InnerProduct(n.prev, num_output=num_features, name='linear',
			weight_filler={'type': 'gaussian', 'std': 0.001,})
	else:
		n.prev = L.InnerProduct(n.input_features, num_output=num_features, name='linear',
			weight_filler={'type': 'gaussian', 'std': 0.001,})


	# caffe will automatically insert split layers when two or more layers have the same bottom, but
	# in so doing, it mangles the name.  By explicitly doing the split, we control the names so that 
	# the blob values can be accessed by the names given here
	n.reconstruction = L.ReLU(n.prev, name='reconstruction')  # assumes that target_features are rectified
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
	s.clip_gradients = 10

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


def extract_from_model(model, num_instances, offset):
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
	reconstructed_features, predicted_output_probs, predicted_classifications = extract_from_model(model, num_instances, offset)
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


def init_empty_metrics():
	d = dict()
	for split in ['train', 'test']:
		d[split] = dict()
		for model_type in ['linear', 'mlp']:
			d[split][model_type] = dict()
			for loss in ['l2', 'ce_soft', 'ce_hard']:
				d[split][model_type][loss] = dict()
	return d


def train_model(model_type, loss, classifier_weights, classifier_bias, num_train_instances, 
		num_test_instances, num_features, num_classes, args):

	net_param = equivariance_proto(args, num_features, num_classes, loss, model_type == 'mlp')
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


def perform_experiment(model_type, loss, classification_weights, classification_bias, num_train_instances, 
		num_test_instances, num_features, num_classes,  args):

	model_train, model_test, train_offset = train_model(model_type, loss, 
				classification_weights, classification_bias, num_train_instances, 
				num_test_instances, num_features, num_classes, args)
	
	with h5py.File(args.train_db, 'r') as f:
		train_metrics = score_model(model_train, f['target_features'], f['target_output_probs'], 
				np.argmax(f['target_output_probs'], axis=1), f['labels'], num_train_instances, train_offset)

	with h5py.File(args.test_db, 'r') as f:
		test_metrics = score_model(model_test, f['target_features'], f['target_output_probs'], 
				np.argmax(f['target_output_probs'], axis=1), f['labels'], num_test_instances, 0)

	return train_metrics, test_metrics


def setup_hdf5s(args):
	with h5py.File(args.train_db, 'w') as train_db:
		with h5py.File(args.input_train_hdf5, 'r') as input_db:
			with h5py.File(args.output_train_hdf5, 'r') as output_db:
				train_db['input_features'] = np.asarray(input_db[args.blob])
				log(args, "Train Input Features Shape: %s" % str(train_db['input_features'].shape))

				train_db['target_features'] = np.asarray(output_db[args.blob])
				log(args, "Train Target Features Shape: %s" % str(train_db['target_features'].shape))

				train_db['target_output_probs'] = np.asarray(output_db['prob'])
				log(args, "Train Target Output Probs Shape: %s" % str(train_db['target_output_probs'].shape))

				train_db['labels'] = np.asarray(output_db['labels'])
				log(args, "Train labels Shape: %s" % str(train_db['labels'].shape))

				num_train_instances, num_features = input_db[args.blob].shape
				num_classes = output_db['prob'].shape[1]

	with h5py.File(args.test_db, 'w') as test_db:
		with h5py.File(args.input_test_hdf5, 'r') as input_db:
			with h5py.File(args.output_test_hdf5, 'r') as output_db:
				test_db['input_features'] = np.asarray(input_db[args.blob])
				log(args, "Test Input Features Shape: %s" % str(test_db['input_features'].shape))

				test_db['target_features'] = np.asarray(output_db[args.blob])
				log(args, "Test Target Features Shape: %s" % str(test_db['target_features'].shape))

				test_db['target_output_probs'] = np.asarray(output_db['prob'])
				log(args, "Test Target Output Probs Shape: %s" % str(test_db['target_output_probs'].shape))

				test_db['labels'] = np.asarray(output_db['labels'])
				log(args, "Test labels Shape: %s" % str(test_db['labels'].shape))

				num_test_instances = input_db[args.blob].shape[0]

	return num_train_instances, num_test_instances, num_features, num_classes
		
		

def main(args):
	log(args, str(args))

	log(args, "Setting up Scratch Space")
	setup_scratch_space(args)

	# pull the classification weights and bias
	log(args, "Loading 'to' model")
	model = init_model(args.network_file, args.weight_file, gpu=args.gpu)

	log(args, "Extracting Classification Weights")
	last_layer_params = model.params.items()[-1][1]
	classification_weights = last_layer_params[0].data
	classification_bias = last_layer_params[1].data

	all_metrics = init_empty_metrics()

	log(args, "Setting up Train/Test DBs")
	num_train_instances, num_test_instances, num_features, num_classes = setup_hdf5s(args)
	log(args, "Num Train Instances: %d" % num_train_instances)
	log(args, "Num Test Instances: %d" % num_test_instances)
	log(args, "Num Features: %d" % num_features)
	log(args, "Num Classes: %d" % num_classes)
	exit()

	log(args, "Starting on Experiments")
	for model_type in ['linear', 'mlp']:
		for loss in ['l2', 'ce_soft', 'ce_hard']:
			log(args, "EXPERIMENT %s %s" % (model_type, loss))
			train_metrics, test_metrics = perform_experiment(model_type, loss, 
				classification_weights, classification_bias, num_train_instances, num_test_instances, 
				num_features, num_classes, args)
			all_metrics['train'][model_type][loss] = train_metrics
			all_metrics['test'][model_type][loss] = test_metrics

	with open(args.out_file, 'w') as f:
		f.write(pprint.pformat(all_metrics))

	#cleanup_scratch_space(args)
	log(args, "Exiting...")
		
	if args.log_file:
		args.log.close()


def get_args():
	parser = argparse.ArgumentParser(
		description="Measures invariance of learned representations with respect to the given transforms")
	parser.add_argument("input_train_hdf5", 
				help="HDF5 of vectors used to train equivalence mappings")
	parser.add_argument("input_test_hdf5", 
				help="HDF5 of vectors used to train equivalence mappings")
	parser.add_argument("output_train_hdf5", 
				help="HDF5 of vectors used to train equivalence mappings")
	parser.add_argument("output_test_hdf5", 
				help="HDF5 of vectors used to train equivalence mappings")
	parser.add_argument("network_file", 
				help="Caffe network file for output representation")
	parser.add_argument("weight_file", 
				help="Caffe weights file for output representation")
	parser.add_argument("out_file", type=str,
				help="File to write the output")

	parser.add_argument("-f", "--log-file", type=str, default="",
				help="Log File")

	parser.add_argument("-e", "--max-epochs", type=int, default=20,
				help="Max training epochs for equivalence models")
	parser.add_argument("-l", "--learning-rate", type=float, default=0.1,
				help="Initial Learning rate for equivalence models")
	parser.add_argument("-k", "--hidden-size", type=float, default=1.,
				help="Fraction of # inputs to determine hidden size for mlp equivalence mappings")
	parser.add_argument("--gpu", type=int, default=0,
				help="GPU to use for running the models")
	parser.add_argument("--blob", type=str, default="InnerProduct2",
				help="Name of db on which to measure equivalence")

	args = parser.parse_args()
	args.train_batch_size = 1

	return args
	

if __name__ == "__main__":
	args = get_args()
	main(args)

