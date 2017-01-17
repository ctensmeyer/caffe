
import os
import sys
import caffe
import cv2
import math
import lmdb
import random
import argparse
import numpy as np
import caffe.proto.caffe_pb2
import scipy.ndimage
import traceback
import errno


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


def get_image(cursor):
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
	means = np.asarray(map(float, means.split(',')))
	means = means[np.newaxis,np.newaxis,np.newaxis,:]
	return float(scale) * (im - means)


def get_batch(cursor, batch_size=64, means=0, scale=1):
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


def fprop(model, ims, args):
	# batch up all transforms at once
	model.blobs[args.input_blob].reshape(len(ims), ims[0].shape[3], ims[0].shape[1], ims[0].shape[2]) 
	for x in xrange(ims.shape[0]):
		transposed = np.transpose(ims[0], [0,3,1,2])
		model.blobs["data"].data[x,:,:,:] = transposed
	# propagate on batch
	model.forward()


def safe_mkdir(_dir):
	try:
		os.makedirs(os.path.join(_dir))
	except OSError as exc:
		if exc.errno != errno.EEXIST:
			raise


def init_fds(args, blobs):
	fds = dict()
	safe_mkdir(args.out_dir)

	for blob in blobs:
		fds[blob] = open(os.path.join(args.out_dir, blob + ".txt"), 'w')
	return fds


def record_activation(model, blob, fd, args):
	activations = model.blobs[blob].data
	if activations.ndim > 2:
		# pool over spatial regions
		#activations = np.max(activations, axis=(2,3))
		activations = activations.reshape((activations.shape[0], -1))
	#print blob, activations.shape
	np.savetxt(fd, activations, "%7.4f")


def main(args):
	print args
	print "Initiaializing Model..."
	model = init_model(args.network_file, args.weight_file, gpu=args.gpu)
	print "Opening LMDB..."
	env, txn, cursor = open_lmdb(args.test_lmdb)
	max_images = min(args.max_images, env.stat()['entries'])
	max_iters = (max_images + args.batch_size - 1) / args.batch_size

	if args.blobs == "_all":
		blobs = model.blobs.keys()
	else:
		blobs = args.blobs.split(',')
	print "Recording blobs:", blobs
	print "Opening Output Files..."
	fds = init_fds(args, blobs)
	label_fd = open(os.path.join(args.out_dir, "labels.txt"), 'w')

	print "Starting Activation Extraction..."
	for iter_num in xrange(max_iters):
		ims, labels = get_batch(cursor, batch_size=args.batch_size, means=args.means, scale=args.scale)
		fprop(model, ims, args)
		for blob in blobs:
			record_activation(model, blob, fds[blob], args)
		for label in labels:
			label_fd.write("%d\n" % label)

		if iter_num > 0 and iter_num % 10 == 0:
			print "%.2f%% (%d/%d) Batches" % (100. * iter_num / max_iters, iter_num, max_iters)
	print "Done"

	print "Closing Files..."
	label_fd.close()
	for fd in fds.values():
		fd.close()

	print "Closing LMDB..."
	env.close()
	print "Exiting"


def get_args():
	parser = argparse.ArgumentParser(
		description="Dumps neuron activations to files")
	parser.add_argument("network_file", 
				help="Caffe network file")
	parser.add_argument("weight_file", 
				help="Caffe weights file")
	parser.add_argument("test_lmdb", 
				help="LMDB of images (encoded DocDatums), used to gather activation values")
	parser.add_argument("out_dir",
				help="Output directory of where to store the activations")

	parser.add_argument("-m", "--means", type=str, default="0",
				help="Optional mean values per channel " 
				"(e.g. 127 for grayscale or 182,192,112 for BGR)")
	parser.add_argument("-a", "--scale", type=str, default=str(1.0 / 255),
				help="Optional scale factor")
	parser.add_argument("--max-images", default=40000, type=int, 
				help="Max number of images for processing")
	parser.add_argument("-b", "--batch-size", default=64, type=int, 
				help="Max number of image processed at once")
	parser.add_argument("--gpu", type=int, default=0,
				help="GPU to use for running the models")
	parser.add_argument("--input-blob", type=str, default="data",
				help="Name of input blob")
	parser.add_argument("--blobs", type=str, default="_all",
				help="Comma separated list of blobs to include")

	args = parser.parse_args()

	return args
	

if __name__ == "__main__":
	args = get_args()
	main(args)
