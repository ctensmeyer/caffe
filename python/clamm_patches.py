
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import caffe
import cv2
import math
import lmdb
import random
import argparse
import StringIO
import numpy as np
import caffe.proto.caffe_pb2

np.set_printoptions(precision=2, linewidth=170, suppress=True)

def init_caffe(args):
	if args.gpu >= 0:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()

	caffenet = caffe.Net(args.caffe_model, args.caffe_weights, caffe.TEST)
	return caffenet

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

def apply_dense_crop(im, tokens):
	height, width, = int(tokens[1]), int(tokens[2])
	y_stride, x_stride, = int(tokens[3]), int(tokens[4])
	if (height > im.shape[0]) or (width > im.shape[1]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, tokens)
		exit(1)
	ims = list()
	weights = list()
	y = 0
	while (y + height) <= im.shape[0]:
		x = 0
		while (x + width) <= im.shape[1]:
			ims.append(im[y:y+height,x:x+width])
			x += x_stride
			if x > 0 and y > 0 and (x + width) <= im.shape[1] and (y + y_stride + height) < im.shape[0]:
				weights.append(2)
			else:
				weights.append(1)
		y += y_stride
	
	return ims, weights

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
	size = int(tokens[1]), int(tokens[2])
	return cv2.resize(im, size)

# "resize2 scale_factor"
def apply_resize2(im, tokens):
	scale_factor = float(tokens[1])
	new_height = int(scale_factor * im.shape[0])
	new_width = int(scale_factor * im.shape[1])
	return cv2.resize(im, (new_height, new_width))

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

# "guassnoise seed sigma"
def apply_gaussnoise(im, tokens):
	seed, sigma = int(tokens[1]), float(tokens[2])
	np.random.seed(seed)
	noise = np.random.normal(0, sigma, im.shape[:2])
	if len(im.shape) == 2:
		im = (im + noise),
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
	if tokens[0] == 'densecrop':
		return apply_dense_crop(im, tokens)
	if tokens[0] == 'randcrop':
		return apply_rand_crop(im, tokens)
	elif tokens[0] == 'resize':
		return apply_resize(im, tokens)
	elif tokens[0] == 'resize2':
		return apply_resize2(im, tokens)
	elif tokens[0] == 'mirror':
		return apply_mirror(im, tokens)
	elif tokens[0] == 'gaussnoise':
		return apply_gaussnoise(im, tokens)
	elif tokens[0] == 'rotation':
		return apply_rotation(im, tokens)
	elif tokens[0] == 'blur':
		return apply_blur(im, tokens)
	elif tokens[0] == 'unsharpmask':
		return apply_unsharpmask(im, tokens)
	elif tokens[0] == 'shear':
		return apply_shear(im, tokens)
	elif tokens[0] == 'perspective':
		return apply_perspective(im, tokens)
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
	weights = list()
	for ts in transform_strs:
		im_out = apply_transforms(im, ts)
		if type(im_out) is list:
			ims.extend(im_out)
			weights.extend([1] * len(im_out))
		elif type(im_out) is tuple:
			ims.extend(im_out[0])
			weights.extend(im_out[1])
		else:
			ims.append(im_out)
			weights.append(1)
	return ims, weights

def get_transforms(args):
	transforms = list()
	scale_tokens = args.resize_scales.split(args.delimiter)
	scale_min, scale_max = float(scale_tokens[0]), float(scale_tokens[1])
	for x in xrange(args.batch_size):
		if scale_min == scale_max:
			scale = scale_min
		else:
			scale = random.uniform(scale_min, scale_max)
		scale_transform = "resize2 %f" % scale
		crop_transform = "randcrop %d %d" % (args.size, args.size)
		transform = ";".join([scale_transform, crop_transform])
		transforms.append(transform)
	return transforms


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

def get_vote_for_label(ims, caffenet, label, args):
	# batch up all transforms at once
	all_outputs = fprop(caffenet, ims, args.batch_size)
	return all_outputs[:, label]


def fprop(caffenet, ims, batchsize=64):
	# batch up all transforms at once
	idx = 0
	responses = list()
	while idx < len(ims):
		sub_ims = ims[idx:idx+batchsize]
		caffenet.blobs["data"].reshape(len(sub_ims), ims[0].shape[2], ims[0].shape[0], ims[0].shape[1]) 
		for x, im in enumerate(sub_ims):
			transposed = np.transpose(im, [2,0,1])
			transposed = transposed[np.newaxis, :, :, :]
			caffenet.blobs["data"].data[x,:,:,:] = transposed
		idx += batchsize

		# propagate on batch
		caffenet.forward()
		responses.append(np.copy(caffenet.blobs["prob"].data))
	return np.concatenate(responses, axis=0)
	
def open_dbs(db_paths, write=False):
	dbs = list()
	for path in db_paths:
		env = lmdb.open(path, readonly=(not write), map_size=int(2 ** 42))
		txn = env.begin(write=write)
		cursor = txn.cursor()
		cursor.first()
		dbs.append( (env, txn, cursor) )
	return dbs

def close_dbs(dbs):
	for env, txn, cursor in dbs:
		txn.commit()
		env.sync()
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
	original_slice_transforms = list()
	labels = list()
	keys = list()

	# apply the transformations to every slice of the image independently
	for slice_idx, entry in enumerate(test_dbs):
		env, txn, cursor = entry

		im_slice, label_slice = get_image(cursor.value(), slice_idx, args)
		transformed_slices, weights = apply_all_transforms(im_slice, transforms)
		original_slice_transforms.append(map(lambda im: np.copy(im), transformed_slices))

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

	return ims, original_slice_transforms, label

def package(im, prob):
	doc_datum = caffe.proto.caffe_pb2.DocumentDatum()
	doc_datum.decade = prob
	datum_im = doc_datum.image

	datum_im.channels = im.shape[2] if len(im.shape) == 3 else 1
	datum_im.width = im.shape[1]
	datum_im.height = im.shape[0]
	datum_im.encoding = 'jpeg'

	# image data
	buf = StringIO.StringIO()
	if datum_im.channels == 1:
		plt.imsave(buf, im, format='jpeg', vmin=0, vmax=256, cmap='gray')
	else:
		# do BGR -> RGB
		im = im[(2, 1, 0), :, :]
		plt.imsave(buf, im, format='jpeg', vmin=0, vmax=256)
	datum_im.data = buf.getvalue()

	return doc_datum

def write_patches(ims, prob_of_label, out_dbs):
	num_slices = len(out_dbs)
	num_patches = len(ims[0])
	for slice_idx in xrange(num_slices):
		for patch_idx in xrange(num_patches):
			patch = ims[slice_idx][patch_idx]
			prob = float(prob_of_label[patch_idx])

			doc_datum = package(patch, prob)

			out_txn = out_dbs[slice_idx][1]
			key = str(patch_idx + 1000 * random.randint(0,100000000))
			out_txn.put(key, doc_datum.SerializeToString())


def main(args):
	log(args, str(sys.argv))

	log(args, "Initializing network for testing")
	caffenet = init_caffe(args)
	log(args, "Opening test lmdbs")
	test_dbs = open_dbs(args.test_lmdbs.split(args.delimiter))
	out_dbs = open_dbs(args.out_lmdbs.split(args.delimiter), write=True)

	num_total = 0
	while num_total < args.num_patches:
		transforms = get_transforms(args)
		if num_total % args.print_count == 0:
			print "Processed %d patches" % num_total
		num_total += len(transforms)

		preprocessed_ims, ims, label = prepare_images(test_dbs, transforms, args)
		prob_of_label = get_vote_for_label(preprocessed_ims, caffenet, label, args)

		# make sure to get exactly args.num_patches
		if num_total + len(ims) > args.num_patches:
			ims = ims[:args.num_patches - num_total]
			prob_of_label = prob_of_label[:args.num_patches - num_total]

		write_patches(ims, prob_of_label, out_dbs)

		for env, txn, cursor in test_dbs:
			has_next = cursor.next() 
			if not has_next:
				cursor.first()

	log(args, "Done")

	close_dbs(test_dbs)
	close_dbs(out_dbs)
		

def check_args(args):
	num_out_lmdbs = 0 if args.out_lmdbs == "" else len(args.out_lmdbs.split(args.delimiter))
	num_test_lmdbs = 0 if args.test_lmdbs == "" else len(args.test_lmdbs.split(args.delimiter))
	if num_test_lmdbs == 0:
		raise Exception("No test lmdbs specified");
	if num_out_lmdbs != num_test_lmdbs:
		raise Exception("Different number of out and test lmdbs: %d vs %d" % (num_out_lmdb, num_test_lmdbs))

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
				help="LMDBs of test images (encoded DocDatums), files separated with ;")
	parser.add_argument("out_lmdbs", 
				help="LMDBs of test images (encoded DocDatums), files separated with ;")

	parser.add_argument("-m", "--means", type=str, default="",
				help="Optional mean values per the channel (e.g. 127 for grayscale or 182,192,112 for BGR)")
	parser.add_argument("--gpu", type=int, default=-1,
				help="GPU to use for running the network")
	parser.add_argument('-c', '--channels', default="0", type=str,
				help='Number of channels to take from each slice')
	parser.add_argument("-a", "--scales", type=str, default=str(1.0 / 255),
				help="Optional scale factor")
	parser.add_argument("-s", "--size", type=int, default=227,
				help="Size of Crops")
	parser.add_argument("--resize-scales", type=str, default="1:1",
				help="Size of Crops")
	parser.add_argument("-f", "--log-file", type=str, default="",
				help="Log File")
	parser.add_argument("--print-count", default=1000, type=int, 
				help="Print every print-count images processed")
	parser.add_argument("--num-patches", default=200000, type=int, 
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

