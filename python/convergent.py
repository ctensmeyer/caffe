
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


def init_model(network_file, weights_file, gpu=0):
	if args.gpu >= 0:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()

	caffenet = caffe.Net(network_file, weights_file, caffe.TEST)
	return caffenet


def read_models(models_manifest, gpu=0):
	models = list()
	for line in open(models_manifest.readlines()):
		tokens = line.split()
		network_file = tokens[0]
		weights_file = tokens[1]

		model = init_model(network_file, weights_file, gpu)
		models.append(model)
	return models


def read_corresponding_layers(layers_manifest, num_models):
	corresponding_layers = list()
	for ln, line in enumerate(open(layers_manifest).readlines()):
		layers = line.split()
		if len(layers) != num_models:
			raise Exception("Found %d layers on line %d of %s.  Expected %d" % (
				len(layers), ln, layers_manifest, num_models))
		corresponding_layers.append(layers)
	return corresponding_layers


def open_lmdb(test_lmdb):
	env = test_lmdb.open(path, readonly=True, map_size=int(2 ** 42))
	txn = env.begin(write=False)
	cursor = txn.cursor()
	cursor.first()
	return env, txn, cursor


def get_image(cursor):
	doc_datum = caffe.proto.caffe_pb2.DocumentDatum()
	doc_datum.ParseFromString(dd_serialized)  # decode lmdb entry

	nparr = np.fromstring(doc_datum.image.data, np.uint8)
	im = cv2.imdecode(nparr, -1)  # load the image as is (do not force color/grayscale)
	if im.ndim == 2:
		# explicit single channel to match dimensions of color
		im = im[:,:,np.newaxis]


def preprocess_im(im, means, scale):
	means = np.asarray(means)
	return scale * (im - means)


def get_batch(cursor, batch_size=64, means=0, scale=1):
	# get first image to determine spatial sizes
	im = get_image(cursor)
	im = preprocess_im(im, means, scale)

	batch_shape = (batch_size,) + im.shape
	ims = np.empty(shape=batch_shape)
	ims[0] = im

	for idx in xrange(1, batch_size):
		im = get_image(cursor)
		im = preprocess_im(im, means, scale)
		ims[idx] = im

	return ims

	


def compute_mean_activations(model, test_lmdb, iters, layers, means=0, scale=1, batch_size=64):
	means = dict()
	for layer in layers:
		size = model.blobs[layer].data.shape[1]  # number of neurons or number of filters
		means[layer] = np.zeros(shape)
		
	env, txn, cursor = open_lmdb(test_lmdb)
	for _iter in xrange(iters):
		ims = get_batch(cursor, batch_size, means, scale)
		
	


def get_args():
	parser = argparse.ArgumentParser(description="Classifies data")
	parser.add_argument("models_manifest", 
				help="File listing pairs of Caffe network/weight filepaths"))
	parser.add_argument("layers_manifest", 
				help="File containing blob lists to correlate")
	parser.add_argument("test_lmdb", 
				help="LMDB of images (encoded DocDatums), used to gather activation values")

	parser.add_argument("-m", "--means", type=str, default="",
				help="Optional mean values per channel (e.g. 127 for grayscale or 182,192,112 for BGR)")
	parser.add_argument("-a", "--scales", type=str, default=str(1.0 / 255),
				help="Optional scale factor")
	parser.add_argument("-f", "--log-file", type=str, default="",
				help="Log File")
	parser.add_argument("--max-images", default=40000, type=int, 
				help="Max number of images for processing")
	parser.add_argument("-b", "--batch-size", default=64, type=int, 
				help="Max number of image processed at once")
	parser.add_argument("--gpu", type=int, default=0,
				help="GPU to use for running the models")

	args = parser.parse_args()

	check_args(args)
	return args
	

if __name__ == "__main__":
	args = get_args()
	main(args)
