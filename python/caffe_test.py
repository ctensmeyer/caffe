
import os
import sys
import caffe
import numpy as np
import cv2
import argparse
import lmdb
import caffe.proto.caffe_pb2

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
		
	return im[y:y+height,x:x+width,:]

# "resize height width"
def apply_resize(im, tokens):
	size = int(tokens[1]), int(tokens[2])
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

# "guassnoise seed sigma"
def apply_gaussnoise(im, tokens):
	seed, sigma = int(tokens[1]), float(tokens[2])
	np.random.seed(seed)
	return im + np.random.normal(0, sigma, im.shape)

# "rotate degree"
def apply_rotation(im, tokens):
	degree = float(tokens[1])
	center = (im.shape[0] / 2, im.shape[1] / 2)
	rot_mat = cv2.getRotationMatrix2D(center, degree, 1.0)
	return cv2.warpAffine(im, rot_mat, im.shape, flags=cv2.INTER_LINEAR)

# "blur sigma"
def apply_blur(im, tokens):
	sigma = float(tokens[1])
	size = int(sigma * 4 + .999)
	if size % 2 == 0:
		size += 1
	return cv2.GaussianBlur(im, (size, size), sigma)
	
# "unsharpmask sigma amount"
def apply_unsharpmask(im, tokens):
	blurred = apply_blur(im)
	return (1 + amount) * im + (-amount * blurred)

# "shear degree {h,v}"
def apply_shear(im, tokens):
	degree = float(tokens[1])
	radians = math.tan(degree * math.pi / 180)
	shear_mat = np.array([ [1, 0, 0], [0, 1, 0] ])
	if tokens[2] == 'h':
		shear_mat[0,1] = radians
	elif tokens[2] == 'v':
		shear_mat[1,0] = radians
	else:
		print "Invalid shear type: %r" % tokens
	return cv2.warpAffine(im, shear_mat, im.shape[:2], flags=cv2.INTER_LINEAR)

# "perspective dy1 dx1 dy2 dx2 dy3 dx3 dy4 dx4"
def apply_perspective(im, tokens):
	pts1 = np.float32([[0,0],[1,0],[1,1],[0,1]])
	pts2 = np.float32([[0 + float(tokens[1]) ,0 + float(tokens[2])],
					   [1 + float(tokens[3]) ,0 + float(tokens[4])],
					   [1 + float(tokens[5]) ,1 + float(tokens[6])],
					   [0 + float(tokens[7]) ,1 + float(tokens[8])]
					   ])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	return cv2.warpPerspective(im, M, im.shape[:2])

def apply_transform(im, transform_str):
	tokens = transform_str.split()
	if tokens[0] == 'crop':
		return apply_crop(im, tokens)
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
	elif tokens[0] == 'unsharpmask':
		return apply_unsharpmask(im, tokens)
	elif tokens[0] == 'shear':
		return apply_shear(im, tokens)
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

def get_transforms(args):
	transforms = list()
	if args.transform_file:
		transforms = map(lambda s: s.rstrip(), open(args.transform_file, 'r').readlines())
	if not transforms:
		transforms.append("none")
	return transforms

def get_image(dd_serialized, args):
	doc_datum = caffe.proto.caffe_pb2.DocumentDatum()
	doc_datum.ParseFromString(dd_serialized)	
	nparr = np.fromstring(doc_datum.image.data, np.uint8)
	im = cv2.imdecode(nparr, 0 if args.gray else 1)
	if args.gray:
		# explicit single channel to match dimensions of color
		im = im[:,:,np.newaxis]
	label = doc_datum.dbid
	return im, label

def preprocess_im(im, args):
	# currently only does mean and shift
	# transposition is handled by predict() so intermediate augmentations can take place
	#im = cv2.resize(im, (227,227))
	mean_vals = np.asarray(map(int, args.means.split(',')))
	return args.scale * (im - mean_vals)

def set_transform_weights(args):
	caffenet = init_caffe(args)
	tune_env = lmdb.open(args.tune_lmdb, readonly=True, map_size=int(2 ** 42))
	txn = tune_env.begin(write=False)
	cursor = txn.cursor()
	transforms = get_transforms(args)

	weights = np.zeros(shape=(len(transforms),))

	num_total = 0
	for key, value in cursor:
		if num_total % 1000 == 0:
			print "Tuned on %d images" % num_total
		num_total += 1

		im, label = get_image(value, args)
		im = preprocess_im(im, args)
		ims = map(lambda transform_strs: apply_transforms(im, transform_strs), transforms)
		votes = get_vote_for_label(ims, caffenet, label)
		#print "Votes: ", votes
		weights += votes

		if num_total == 40000:
			break

	normalized = (weights / num_total)[:,np.newaxis]
	return normalized

def get_vote_for_label(ims, caffenet, label):
	# batch up all transforms at once
	caffenet.blobs["data"].reshape(len(ims), ims[0].shape[2], ims[0].shape[0], ims[0].shape[1]) 
	for idx, im in enumerate(ims):
		transposed = np.transpose(im, [2,0,1])
		transposed = transposed[np.newaxis, :, :, :]
		caffenet.blobs["data"].data[idx,:,:,:] = transposed

	# propagate on all transforms
	caffenet.forward()

	# sum up weighted prediction vectors
	all_outputs = caffenet.blobs["prob"].data
	return all_outputs[:, label]

def predict(ims, caffenet, weights=None):
	# set up transform weights
	if weights is None:
		weights = np.array([1] * len(ims))
		weights = weights[:,np.newaxis]
		
	# batch up all transforms at once
	caffenet.blobs["data"].reshape(len(ims), ims[0].shape[2], ims[0].shape[0], ims[0].shape[1]) 
	for idx, im in enumerate(ims):
		transposed = np.transpose(im, [2,0,1])
		transposed = transposed[np.newaxis, :, :, :]
		caffenet.blobs["data"].data[idx,:,:,:] = transposed

	# propagate on all transforms
	caffenet.forward()

	# sum up weighted prediction vectors
	all_outputs = caffenet.blobs["prob"].data
	weighted_outputs = all_outputs * weights
	mean_outputs = np.sum(all_outputs, axis=0)
	label = np.argmax(mean_outputs)
	return label

def main(args):
	if args.log_file:
		log = open(args.log_file, 'w')
		log.write("%s\n" % str(sys.argv))
	caffenet = init_caffe(args)
	test_env = lmdb.open(args.test_lmdb, readonly=True, map_size=int(2 ** 42))
	txn = test_env.begin(write=False)
	cursor = txn.cursor()
	transforms = get_transforms(args)
	weights = None
	if args.tune_lmdb:
		weights = set_transform_weights(args)
	#print "Weights: %r" % weights
	if args.log_file:
		log.write("%r\n" % weights)

	num_total = 0
	num_correct = 0
	for key, value in cursor:
		if num_total % 1000 == 0:
			print "Processed %d images" % num_total
		num_total += 1

		im, label = get_image(value, args)
		im = preprocess_im(im, args)
		ims = map(lambda transform_strs: apply_transforms(im, transform_strs), transforms)

		predicted_label = predict(ims, caffenet, weights)
		if predicted_label == label:
			num_correct += 1

	acc = (100. * num_correct / num_total)
	print "Done"
	print "Accuracy: %.3f%%" % acc
	if args.log_file:
		log.write("%f\n" % acc)
			
def get_args():
	parser = argparse.ArgumentParser(description="Classifies data")
	parser.add_argument("caffe_model", 
				help="The model definition file (e.g. deploy.prototxt)")
	parser.add_argument("caffe_weights", 
				help="The model weight file (e.g. net.caffemodel)")
	parser.add_argument("test_lmdb", 
				help="LMDB of test images (encoded DocDatums)")

	parser.add_argument("-m", "--means", type=str, default="",
				help="Optional mean values per the channel (e.g. 127 for grayscale or 182,192,112 for BGR)")
	parser.add_argument("--gpu", type=int, default=-1,
				help="GPU to use for running the network")
	parser.add_argument('-g', '--gray', default=False, action="store_true",
						help='Force images to be grayscale.  Force color if ommited')
	parser.add_argument("-a", "--scale", type=float, default=(1.0 / 255),
				help="Optional scale factor")
	parser.add_argument("-t", "--transform_file", type=str, default="",
				help="File containing transformations to do")
	parser.add_argument("-l", "--tune-lmdb", type=str, default="",
				help="Tune the weighted averaging to minmize CE loss on this data")
	parser.add_argument("-f", "--log-file", type=str, default="",
				help="Log File")
	
	return parser.parse_args()
	

if __name__ == "__main__":
	args = get_args()
	main(args)

