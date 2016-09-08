
import os
import sys
import cv2
import dpp
import lmdb
import caffe
import shutil
import argparse
import tempfile
import numpy as np
import caffe.proto.caffe_pb2
from sklearn import linear_model
import google.protobuf.text_format
from sklearn.metrics.pairwise import pairwise_kernels



def log(args, s, newline=True):
	print s
	if args.log_file:
		if not hasattr(args, 'log'):
			args.log = open(args.log_file, 'w')
		if newline:
			args.log.write("%s\n" % s)
		else:
			args.log.write(s)

def init_caffe(args):
	if args.gpu >= 0:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()

	model = caffe.Net(args.network_file, args.weight_file, caffe.TEST)
	return model


def gram(mat, args):
	'''Computes the gram matrix of mat according to a specified kernel function'''
	kwargs = {}

	if args.kernel in ['rbf', 'polynomial', 'poly', 'laplacian']:
		# gamma for chi squared should be left to default
		kwargs = dict(gamma=10. / mat_a.shape[1])
	output = pairwise_kernels(mat, metric=args.kernel, n_jobs=1, **kwargs)
	return output


def sample_neurons(L, args):
	return dpp.sample_dpp(L, args.num_neurons)


def get_layer(layer_name, model):
	for layer, name in zip(model.layers, model._layer_names):
		if name == layer_name:
			return layer
	else:
		return None


def load_net_spec(args):
	net_spec = caffe.proto.caffe_pb2.NetParameter()
	_str = open(args.network_file, 'r').read()
	google.protobuf.text_format.Merge(_str, net_spec)
	return net_spec


def save_net_spec(netspec, args):
	fd = open(args.out_network_file, 'w')
	fd.write(str(netspec))
	fd.close()


def get_layer_param(layer_name, netspec):
	for layer_param in netspec.layer:
		if layer_param.name == layer_name:
			return layer_param
	else:
		return None


def update_weights(inputs, keep, output):
	inputs = inputs[:,keep]
	clf = linear_model.LinearRegression(fit_intercept=True)
	clf.fit(inputs, output)
	weights = clf.coef_
	intercept = clf.intercept_

	return weights, intercept


def safe_mkdir(_dir):
	try:
		os.makedirs(os.path.join(_dir))
	except OSError as exc:
		if exc.errno != errno.EEXIST:
			raise


def set_up_tmp_dirs():
	tmp_dir = tempfile.mkdtemp()
	original_activations_dir = os.path.join(tmp_dir, "original_activations")
	safe_mkdir(original_activations_dir)
	current_activations_dir = os.path.join(tmp_dir, "current_activations")
	safe_mkdir(current_activations_dir)
	return original_activations_dir, current_activations_dir


def open_lmdb(lmdb_path):
	env = lmdb.open(lmdb_path, readonly=True, map_size=int(2 ** 38))
	txn = env.begin(write=False)
	cursor = txn.cursor()
	cursor.first()
	return env, txn, cursor


def init_fds(out_dir, blobs):
	'''
	Opens file descriptors for each blob in out_dir
	'''
	fds = dict()

	for blob in blobs:
		fds[blob] = open(os.path.join(out_dir, blob + ".txt"), 'w')
	return fds


def get_image(cursor):
	'''
	Returns a single image from the cursor.  The LMDB is assumed to
		contain entries that are serialized DocumentDatum protobuf
		messages.
	'''
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
	im = cv2.imdecode(nparr, 0)  
	if im.ndim == 2:
		# explicit single channel to match dimensions of color
		im = im[:,:,np.newaxis]

	return im, label


def preprocess_im(im, means, scale):
	'''
	Subtract mean from each channel and scale pixel intensities
	'''
	means = np.asarray(map(float, means.split(',')))
	means = means[np.newaxis,np.newaxis,:]
	return float(scale) * (im - means)


def get_batch(cursor, batch_size=64, means=0, scale=1):
	'''
	Pulls $batch_size images from the LDMB cursor and preprocesses
		them according to mean and scale
	'''
	# get first image to determine spatial sizes
	labels = list()
	im, label = get_image(cursor)
	im = preprocess_im(im, means, scale)
	labels.append(label)

	batch_shape = (batch_size,) + im.shape
	ims = np.empty(shape=batch_shape)
	ims[0] = im


	for idx in xrange(1, batch_size):
		im, label = get_image(cursor)
		if im is None:
			break
		im = preprocess_im(im, means, scale)
		ims[idx] = im
		labels.append(label)

	# stopped short of a full batch
	if idx < (batch_size - 1):
		ims = ims[0:idx]

	return ims, labels


def fprop(model, batch, args):
	'''
	Propagates the images in the batch through model in a single forward pass
	'''
	# batch up all transforms at once
	model.blobs[args.input_blob].reshape(batch.shape[0], batch.shape[3], batch.shape[1], batch.shape[2]) 
	transposed = np.transpose(batch, [0,3,1,2])
	model.blobs["data"].data[:] = transposed
	#for x in xrange(batch.shape[0]):
	#	transposed = np.transpose(batch[x], [0,3,1,2])
	#	model.blobs["data"].data[x,:,:,:] = transposed

	# propagate on batch
	model.forward()


def record_blob_activation(model, blob, fd):
	'''
	Saves the activations of the named blob of the model to the file fd
	Activations are flattened to 2D
	'''
	activations = model.blobs[blob].data
	if activations.ndim > 2:
		# pool over spatial regions
		#activations = np.max(activations, axis=(2,3))
		activations = activations.reshape((activations.shape[0], -1))
	np.savetxt(fd, activations, "%7.4f")


def record_activations(model, out_dir, args):
	'''
	Uses args.lmdb as inputs to the model to record activation vectors for every unique
		layer value (blob) in model.  Note that in place operations reuse the same blob,
		so the recorded activations will be after the final operation on that blob.  It
		is recommended to remove in-place operations in the model prototxt.
	Activations are written to files in out_dir, where the file name is 
		$out_dir/<blob_name>.txt.  This is done so that we don't need to store all
		activations for the entire dataset in RAM (it's big).
	'''
	log(args, "Opening LMDB")
	env, txn, cursor = open_lmdb(args.lmdb)

	# compute the number of batches
	num_images = min(args.max_images, env.stat()['entries'])
	num_batches = (num_images + args.batch_size - 1) / args.batch_size
	log(args, "Will execute for %d batches" % num_batches)

	# get the blob names and open files for each one
	blob_names = model.blobs.keys()
	log(args, "Recording activations for blobs: %r" % blob_names)
	fds = init_fds(out_dir, blob_names)
	label_fd = open(os.path.join(out_dir, "labels.txt"), 'w')
	
	for batch_num in xrange(num_batches):
		batch, labels = get_batch(cursor, batch_size=args.batch_size, means=args.means, scale=args.scale)
		fprop(model, batch, args)
		for blob_name in blob_names:
			record_blob_activation(model, blob_name, fds[blob_name])
		for label in labels:
			label_fd.write("%d\n" % label)

		if batch_num > 0 and batch_num % 10 == 0:
			log(args, "%.2f%% (%d/%d) Batches" % (100. * batch_num / num_batches, batch_num, num_batches))

	log(args, "Done Extracting activations")
	log(args, "Closing Files")
	label_fd.close()
	for fd in fds.values():
		fd.close()

	log(args, "Closing LMDB")
	env.close()


def copy_files(from_dir, to_dir):
	for fn in os.listdir(from_dir):
		from_file = os.path.join(from_dir, fn)
		to_file = os.path.join(to_dir, fn)
		shutil.copy(from_file, to_file)


def get_layer_names(args, model):
	if args.layers == "_all":
		# only include layers with modifiable parameters
		layer_names = [name for (layer, name) in zip(model.layers, model._layer_names) if layer.type == "InnerProduct"]
		del layer_names[0]  # cannot prune input neurons
	else:
		layer_names = args.layers.split(args.delimiter)
	return layer_names

def get_all_ip_layer_names(model):
	return [name for (layer, name) in zip(model.layers, model._layer_names) if layer.type == "InnerProduct"]


def prune_prev_layer(cur_layer_name, neuron_indices_to_keep, model, netspec, args):
	# reshape weights of previous layer
	all_ip_layer_names = get_all_ip_layer_names(model)
	cur_ip_layer_idx = all_ip_layer_names.index(cur_layer_name)
	prev_ip_layer_idx = cur_ip_layer_idx -1
	prev_layer_name = all_ip_layer_names[prev_ip_layer_idx]
	prev_layer = get_layer(prev_layer_name, model)
	log(args, "Prev Layer name: %s" % str(prev_layer_name))

	prev_layer_weights = prev_layer.blobs[0].data
	prev_layer_bias = prev_layer.blobs[1].data
	log(args, "Prev Weight shape: %s" % str(prev_layer_weights.shape))
	log(args, "Prev Bias shape: %s" % str(prev_layer_bias.shape))
	new_prev_layer_weights = prev_layer_weights[neuron_indices_to_keep, :]
	new_prev_layer_bias = prev_layer_bias[neuron_indices_to_keep]
	log(args, "New Prev Weight shape: %s" % str(
		new_prev_layer_weights.shape))
	log(args, "New Prev Bias shape: %s" % str(new_prev_layer_bias.shape))

	prev_layer.blobs[0].reshape(new_prev_layer_weights.shape[0],
								new_prev_layer_weights.shape[1])
	prev_layer.blobs[0].data[:] = new_prev_layer_weights[:]
	prev_layer.blobs[1].reshape(new_prev_layer_bias.shape[0])
	prev_layer.blobs[1].data[:] = new_prev_layer_bias[:]

	# modify num_output for previous layer_param
	prev_layer_param = get_layer_param(prev_layer_name, netspec)
	prev_layer_param.inner_product_param.num_output = (
		neuron_indices_to_keep.shape[0])

	
def prune_layer(layer_name, model, netspec, original_activations_dir, current_activations_dir, args):
	log(args, "Starting to prune Layer %s\n" % layer_name)

	layer = get_layer(layer_name, model)
	log(args, "Old Weight Shape: %s" % str(layer.blobs[0].data.shape))
	log(args, "Old Bias Shape: %s" % str(layer.blobs[1].data.shape))

	layer_param = get_layer_param(layer_name, netspec)
	if layer_param is None:
		raise Exception("Layer %s does not exist in file %s" % (layer_name, args.network_file))
	bottom_blob_name = layer_param.bottom[0]
	bottom_activations_file = os.path.join(current_activations_dir, "%s.txt" % bottom_blob_name)
	bottom_activations = np.loadtxt(bottom_activations_file)
	log(args, "Bottom shape: %s" % str(bottom_activations.shape))

	top_blob_name = layer_param.top[0]
	top_activations_file = os.path.join(original_activations_dir, "%s.txt" % top_blob_name)
	top_activations = np.loadtxt(top_activations_file)
	log(args, "Top shape: %s" % str(top_activations.shape))

	# row = instance, col = neuron, so to get neuron similarity, we transpose
	gram_matrix = gram(bottom_activations.transpose(), args)
	log(args, "Gram Matrix shape: %s" % str(gram_matrix.shape))
	neuron_indices_to_keep = sample_neurons(gram_matrix, args)

	weights, bias = update_weights(bottom_activations, neuron_indices_to_keep, top_activations)
	log(args, "New Weight shape: %s" % str(weights.shape))
	log(args, "New Bias shape: %s" % str(bias.shape))

	layer.blobs[1].data[:] = bias[:]
	layer.blobs[0].reshape(weights.shape[0], weights.shape[1])
	layer.blobs[0].data[:] = weights[:]

	prune_prev_layer(layer_name, neuron_indices_to_keep, model, netspec, args)



def main(args):
	log(args, str(args))

	log(args, "Loading network and weights")
	model = init_caffe(args)

	# We want to reconstruct the original activations from the current activations.
	# The current activations may be different from the original activations if more
	# than one layer is specified.
	log(args, "Creating tmp dirs")
	original_activations_dir, current_activations_dir = set_up_tmp_dirs()

	log(args, "Computing initial activations")
	record_activations(model, original_activations_dir, args)

	# copy over the activations to avoid recomputing them
	copy_files(original_activations_dir, current_activations_dir)

	log(args, "\nParsing network file")
	netspec = load_net_spec(args)

	layer_names = get_layer_names(args, model)
	all_ip_layer_names = get_all_ip_layer_names(model)
	log(args, "Pruning Layers: %r\n" % layer_names)
	first = True
	for layer_name in layer_names:
		if layer_name == all_ip_layer_names[0]:
			log(args, "Skipping Layer %s.  Cannot prune input neurons" % layer_name)
			continue
		if layer_name not in all_ip_layer_names:
			log(args, "Skipping Layer %s.  Not an InnerProduct Layer" % layer_name)
			continue
		if not first:
			record_activations(model, current_activations_dir, args)
		first = False
		prune_layer(layer_name, model, netspec, original_activations_dir, current_activations_dir, args)

	save_net_spec(netspec, args)
	model.save(args.out_weight_file)
		

def get_args():
	parser = argparse.ArgumentParser(description="Classifies data")
	parser.add_argument("network_file", 
				help="The model definition file (e.g. deploy.prototxt)")
	parser.add_argument("weight_file", 
				help="The model weight file (e.g. net.caffemodel)")
	parser.add_argument("lmdb", 
				help="LMDB used to get activations for pruning")
	parser.add_argument("out_network_file", 
				help="Outfile for the pruned caffe architecture")
	parser.add_argument("out_weight_file", 
				help="The output model weight file for the pruned caffe model")

	parser.add_argument("-k", "--num-neurons", type=int, default=0,
				help="Specify the number of neurons to keep at each layer")
	parser.add_argument("-l", "--layers", type=str, default="_all",
				help="delimited list of layer names to prune")
	parser.add_argument("-d", "--delimiter", default=',', type=str, 
				help="Delimiter used for indicating multiple image slice parameters")
	parser.add_argument("-f", "--log-file", type=str, default="",
				help="Log File")
	parser.add_argument("--kernel", type=str, default='linear',
				help="Kernel used to construct Gram Matrix")
	parser.add_argument("-b", "--batch-size", default=64, type=int, 
				help="Max number of images in a batch for activation extraction")
	parser.add_argument("--print-count", default=1000, type=int, 
				help="Print every print-count images processed")
	parser.add_argument("--max-images", default=40000, type=int, 
				help="Max number of images for processing or tuning")
	parser.add_argument("-a", "--scale", type=str, default=str(1.0 / 255),
				help="Optional scale factor")
	parser.add_argument("-m", "--means", type=str, default="0",
				help="Optional mean values per the channel (e.g. 127 for grayscale or 182,192,112 for BGR)")
	parser.add_argument("-n", "--normalize", default=False, action="store_true",
				help="Normalize activation vectors for DPP kernel computation (subtract mean, divide by std dev)")
	parser.add_argument("--gpu", type=int, default=-1,
				help="GPU to use for running the network")
	parser.add_argument("--input-blob", type=str, default="data",
				help="Name of input blob")

	args = parser.parse_args()

	return args
	

if __name__ == "__main__":
	args = get_args()
	main(args)

