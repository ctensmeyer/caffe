
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
import h5py
from utils import safe_mkdir, apply_transform

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

def init_model(network_file, weights_file, gpu=0):
	if args.gpu >= 0:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()

	model = caffe.Net(network_file, weights_file, caffe.TEST)
	return model


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


# currently only does mean and shift
# transposition is handled by predict() so intermediate augmentations can take place
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


# slice index refers to which LMDB the partial image came from
# transform index refers to which transforms of the image
def prepare_image(dbs, args):
	im_slices = list()
	labels = list()
	keys = list()

	for slice_idx, entry in enumerate(dbs):
		env, txn, cursor = entry

		im_slice, label_slice = get_image(cursor.value(), slice_idx, args)
		im_slice = apply_transform(im_slice, args.transform)
		im_slice = scale_shift_im(im_slice, slice_idx, args)

		im_slices.append(im_slice)
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

	whole_im = np.concatenate(im_slices, axis=2) # append along channels

	return whole_im, label

def get_batch(dbs, args):
	ims = list()
	labels = list()

	for _ in xrange(args.batch_size):
		im, label = prepare_image(dbs, args)
		ims.append(im)
		labels.append(label)

		has_next = True
		for entry in dbs:
			env, txn, cursor = entry
			has_next &= cursor.next()

		if not has_next:
			break

	return ims, labels


def fprop(model, ims, args):
	# batch up all transforms at once
	transposed = np.transpose(ims, [0,3,1,2])
	model.blobs[args.input_blob].reshape(*transposed.shape)
	model.blobs[args.input_blob].data[:,:,:,:] = transposed
	model.forward()



def main(args):
	model = init_model(args.network_file, args.weight_file, gpu=args.gpu)
	print args
	dbs = open_dbs(args.lmdbs.split(args.delimiter))
	max_images = min(args.max_images, dbs[0][0].stat()['entries'])
	max_iters = (max_images + args.batch_size - 1) / args.batch_size

	blobs = args.blobs.split(args.delimiter)
	activations = {blob: list() for blob in blobs}
	all_labels = list()

	for iter_num in xrange(max_iters):
		ims, labels = get_batch(dbs, args)

		fprop(model, ims, args)
		for blob in blobs:
			batch_activations = model.blobs[blob].data
			if batch_activations.ndim > 2:
				# pool over spatial regions
				batch_activations = np.max(batch_activations, axis=(2,3))
			activations[blob].append(np.copy(batch_activations))
		all_labels.extend(labels)

		if iter_num > 0 and iter_num % 10 == 0:
			print "%.2f%% (%d/%d) Batches" % (100. * iter_num / max_iters, iter_num, max_iters)

	labels = np.asarray(all_labels, dtype=np.float32)
	print labels.shape
	with h5py.File(args.out_hdf5, 'w') as f:
		f['labels'] = labels
		for blob in blobs:
			arr = np.concatenate(activations[blob], axis=0)
			print arr.shape
			f[blob] = arr

	close_dbs(dbs)


def get_args():
	parser = argparse.ArgumentParser(
		description="Dumps neuron activations to files")

	parser.add_argument("network_file", 
				help="Caffe network file")
	parser.add_argument("weight_file", 
				help="Caffe weights file")
	parser.add_argument("blobs", 
				help="Comma separated list of blobs to include")
	parser.add_argument("lmdbs", 
				help="LMDBs of images (encoded DocDatums), files separated by a delimiter (default :)")
	parser.add_argument("out_hdf5",
				help="Output db of where to store the activations")

	parser.add_argument("-m", "--means", type=str, default="",
				help="Optional mean values per the channel (e.g. 127 for grayscale or 182,192,112 for BGR)")
	parser.add_argument('-c', '--channels', default="0", type=str,
				help='Number of channels to take from each slice')
	parser.add_argument("-a", "--scales", type=str, default=str(1.0 / 255),
				help="Optional scale factor")
	parser.add_argument("--max-images", default=40000, type=int, 
				help="Max number of images for processing")
	parser.add_argument("-b", "--batch-size", default=100, type=int, 
				help="Max number of image processed at once")
	parser.add_argument("--gpu", type=int, default=0,
				help="GPU to use for running the models")
	parser.add_argument("--input-blob", type=str, default="data",
				help="Name of input blob")
	parser.add_argument("--transform", type=str, default="none",
				help="Transform to apply")
	parser.add_argument("-d", "--delimiter", default=':', type=str, 
				help="Delimiter used for indicating multiple image slice parameters")

	args = parser.parse_args()

	return args
	

if __name__ == "__main__":
	args = get_args()
	main(args)
