
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
from utils import get_transforms, apply_all_transforms

def init_caffe(args):
	if args.gpu >= 0:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()

	caffenet = caffe.Net(args.caffe_model, args.caffe_weights, caffe.TEST)
	return caffenet


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

def set_transform_weights(args):
	# check if transform weights need to be done
	if args.tune_lmdbs == "":
		# no lmdb is provided for tuning the weights
		return None
	transforms, fixed_transforms = get_transforms(args)
	if not fixed_transforms:
		# number of transforms varies by image, so no fixed set of weights
		return None

	try:
		caffenet = init_caffe(args)
		tune_dbs = open_dbs(args.tune_lmdbs.split(args.delimiter))

		weights = np.zeros(shape=(len(transforms),))
		num_total = 0
		done = False
		while not done:
			if num_total % args.print_count == 0:
				print "Tuned %d images" % num_total
			num_total += 1

			# get the per-transform vote for the correct label
			ims, label = prepare_images(tune_dbs, transforms, args)
			votes = get_vote_for_label(ims, caffenet, label, args)
			weights += votes

			# check stopping criteria
			done = (num_total == args.max_images)
			for env, txn, cursor in tune_dbs:
				has_next = cursor.next() 
				done |= (not has_next) # set done if there are no more elements

		normalized = (weights / num_total)[:,np.newaxis]
		return normalized
	except Exception as e:
		traceback.print_exc()
		print e
		raise
	finally:
		close_dbs(tune_dbs)

def get_vote_for_label(ims, caffenet, label, args):
	# batch up all transforms at once
	all_outputs = fprop(caffenet, ims, args.batch_size)

	if args.hard_weights:
		# use 1/0 right or not
		predictions = np.argmax(all_outputs, axis=1)
		accuracy = np.zeros(shape=(len(ims),))
		accuracy[predictions == label] = 1
		return accuracy
	else:
		# use the probability of the correct label
		return all_outputs[:, label]


def fprop(caffenet, ims, batchsize=64):
	# batch up all transforms at once
	idx = 0
	responses = list()
	while idx < len(ims):
		sub_ims = ims[idx:idx+batchsize]
		caffenet.blobs["data"].reshape(len(sub_ims), sub_ims[0].shape[2], sub_ims[0].shape[0], sub_ims[0].shape[1]) 
		for x, im in enumerate(sub_ims):
			transposed = np.transpose(im, [2,0,1])
			transposed = transposed[np.newaxis, :, :, :]
			caffenet.blobs["data"].data[x,:,:,:] = transposed
		idx += batchsize

		# propagate on batch
		caffenet.forward()
		responses.append(np.copy(caffenet.blobs["prob"].data))
	return np.concatenate(responses, axis=0)
	

def predict(ims, caffenet, args, weights=None):
	# set up transform weights
	if weights is None:
		weights = np.array([1] * len(ims))
		weights = weights[:,np.newaxis]

	all_outputs = fprop(caffenet, ims, args.batch_size)

	all_predictions = np.argmax(all_outputs, axis=1)
	weighted_outputs = all_outputs * weights
	mean_outputs = np.sum(weighted_outputs, axis=0)
	label = np.argmax(mean_outputs)
	return label, all_predictions

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

def log(args, s, newline=True):
	print s
	if args.log_file:
		if not hasattr(args, 'log'):
			args.log = open(args.log_file, 'w')
		if newline:
			args.log.write("%s\n" % s)
		else:
			args.log.write(s)


# slice index refers to which LMDB the partial image came from
# transform index refers to which transforms of the image
def prepare_images(test_dbs, transforms, args):
	ims_slice_transforms = list()
	labels = list()
	keys = list()

	# apply the transformations to every slice of the image independently
	for slice_idx, entry in enumerate(test_dbs):
		env, txn, cursor = entry

		im_slice, label_slice = get_image(cursor.value(), slice_idx, args)
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


def main(args):
	log(args, str(sys.argv))

	# load transforms from file
	log(args, "Loading transforms")
	transforms, fixed_transforms = get_transforms(args)
	log(args, "Fixed Transforms: %s" % str(fixed_transforms))

	# get per-transform weights.  Can be none if transforms produce variable numbers of images, or
	# no lmdb is provided to tune the weights
	log(args, "Setting the transform weights...")
	weights = set_transform_weights(args) 
	weight_str = np.array_str(weights, max_line_width=80, precision=4) if weights is not None else str(weights)
	log(args, "Weights: %s" % weight_str)

	log(args, "Initializing network for testing")
	caffenet = init_caffe(args)
	log(args, "Opening test lmdbs")
	test_dbs = open_dbs(args.test_lmdbs.split(args.delimiter))

	try:
		# set up the class confusion matrix
		num_output = caffenet.blobs["prob"].data.shape[1]
		conf_mat = np.zeros(shape=(num_output, num_output), dtype=np.int)

		num_total = 0
		num_correct = 0
		all_num_correct = np.zeros(shape=(len(transforms),))
		done = False
		while not done:
			if num_total % args.print_count == 0:
				print "Processed %d images" % num_total
			num_total += 1

			ims, label = prepare_images(test_dbs, transforms, args)
			predicted_label, all_predictions = predict(ims, caffenet, args, weights)

			# keep track of correct predictions
			if predicted_label == label:
				num_correct += 1
			conf_mat[label,predicted_label] += 1

			# compute per-transformation accuracy
			if all_predictions.shape[0] == all_num_correct.shape[0]:
				all_num_correct[all_predictions == label] += 1

			# check stopping criteria
			done = (num_total == args.max_images)
			for env, txn, cursor in test_dbs:
				has_next = cursor.next() 
				done |= (not has_next) # set done if there are no more elements

		overall_acc = float(num_correct) / num_total
		transform_accs = all_num_correct / num_total

		log(args, "Done")
		log(args, "Conf Mat:\n %r" % conf_mat)
		log(args, "\nTransform Accuracy:\n %r" % transform_accs)
		log(args, "\nOverall Accuracy: %f" % overall_acc)
	except Exception as e:
		traceback.print_exc()
		print e
		raise
	finally:
		close_dbs(test_dbs)
		args.log.close()
		

def check_args(args):
	num_tune_lmdbs = 0 if args.tune_lmdbs == "" else len(args.tune_lmdbs.split(args.delimiter))
	num_test_lmdbs = 0 if args.test_lmdbs == "" else len(args.test_lmdbs.split(args.delimiter))
	if num_test_lmdbs == 0:
		raise Exception("No test lmdbs specified");
	if num_tune_lmdbs != 0 and num_tune_lmdbs != num_test_lmdbs:
		raise Exception("Different number of tune and test lmdbs: %d vs %d" % (num_tune_lmdb, num_test_lmdbs))

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
	parser = argparse.ArgumentParser(description="Classifies data")
	parser.add_argument("caffe_model", 
				help="The model definition file (e.g. deploy.prototxt)")
	parser.add_argument("caffe_weights", 
				help="The model weight file (e.g. net.caffemodel)")
	parser.add_argument("test_lmdbs", 
				help="LMDBs of test images (encoded DocDatums), files separated with :")

	parser.add_argument("-m", "--means", type=str, default="",
				help="Optional mean values per the channel (e.g. 127 for grayscale or 182,192,112 for BGR)")
	parser.add_argument("--gpu", type=int, default=-1,
				help="GPU to use for running the network")
	parser.add_argument('-c', '--channels', default="0", type=str,
				help='Number of channels to take from each slice')
	parser.add_argument("-a", "--scales", type=str, default=str(1.0 / 255),
				help="Optional scale factor")
	parser.add_argument("-t", "--transform_file", type=str, default="",
				help="File containing transformations to do")
	parser.add_argument("-l", "--tune-lmdbs", type=str, default="",
				help="Tune the weighted averaging to minmize CE loss on this data")
	parser.add_argument("-f", "--log-file", type=str, default="",
				help="Log File")
	parser.add_argument("-z", "--hard-weights", default=False, action="store_true",
				help="Compute Transform weights using hard assignment")
	parser.add_argument("--print-count", default=1000, type=int, 
				help="Print every print-count images processed")
	parser.add_argument("--max-images", default=40000, type=int, 
				help="Max number of images for processing or tuning")
	parser.add_argument("-d", "--delimiter", default=':', type=str, 
				help="Delimiter used for indicating multiple image slice parameters")
	parser.add_argument("-b", "--batch-size", default=64, type=int, 
				help="Max number of transforms in single batch per original image")

	args = parser.parse_args()

	check_args(args)
	return args
	

if __name__ == "__main__":
	args = get_args()
	main(args)

