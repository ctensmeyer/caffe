import os
import sys
import argparse
import lmdb
import StringIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import caffe
import caffe.proto.caffe_pb2
import random
import json
import traceback
import collections
import pickle
import numpy as np
import skimage.io
import skimage.color
import skimage.filters
import shutil
import cv2

def get_size(size, args):
	longer = max(size)
	shorter = min(size)
	if 'x' in args.size_str:
		out_size = tuple(map(int, args.size_str.split('x')))
	elif args.size_str.endswith('l'):
		s = int(args.size_str[:-1])
		scale = float(s) / longer
		new_shorter = int(max(shorter * scale, s / args.truncate))
		mod = (new_shorter - s) % args.aspect_ratio_bin
		if mod >= args.aspect_ratio_bin / 2:
			new_shorter += (args.aspect_ratio_bin - mod)
		else:
			new_shorter -= mod
		if longer == size[0]:
			out_size = (s, new_shorter)
		else:
			out_size = (new_shorter, s)
	elif args.size_str.endswith('s'):
		s = int(args.size_str[:-1])
		scale = float(s) / shorter
		new_longer = int(min(longer * scale, s * args.truncate))
		mod = (new_longer - s) % args.aspect_ratio_bin
		if mod >= args.aspect_ratio_bin / 2:
			new_longer += (args.aspect_ratio_bin - mod)
		else:
			new_longer -= mod
		if shorter == size[0]:
			out_size = (s, new_longer)
		else:
			out_size = (new_longer, s)
	elif args.size_str.endswith('a'):
		s = int(args.size_str[:-1])
		ar = size[0] / float(size[1])  # height : width ratio
		min_ar = 1. / args.truncate
		max_ar = args.truncate
		if ar < min_ar:
			ar = min_ar
		if ar > max_ar:
			ar = max_ar
		ar = round_ar(ar, min_ar, args.aspect_ratio_bin2)
		out_size = max_size_under_area(ar, s * s)
	else:
		out_size = (int(args.size_str), int(args.size_str))
	return out_size


def max_size_under_area(ar, max_area):
	area = 0
	height = 0
	width = 0
	while area < max_area:
		width += 1
		height = int(ar * width)
		area = width * height
	width -= 1
	height = int(ar * width)
	return (height, width)


def round_ar(ar, min_ar, step):
	new_ar = min_ar
	while new_ar < ar:
		new_ar += step
	if (new_ar - ar) > step / 2:
		return new_ar - step
	else:
		return new_ar
	

def process_im(im_file, args):
	if args.color:
		im = skimage.img_as_ubyte(skimage.io.imread(im_file, as_grey=False))
		if len(im.shape) == 2:
			# image is originally grayscale
			im = skimage.color.gray2rgb(im)
	elif args.gray or args.dense_surf:
		im = skimage.img_as_ubyte(skimage.io.imread(im_file, as_grey=True))
	elif args.binary:
		im = skimage.img_as_ubyte(skimage.io.imread(im_file, as_grey=True))
		binary_thresh = skimage.filters.threshold_otsu(im)
		if binary_thresh == 0:
			binary_thresh = 0.5
		low_vals = im < binary_thresh
		high_vals = im >= binary_thresh
		im[low_vals] = 0
		im[high_vals] = 255
	else:
		raise Exception("Must specify one of {color,gray,binary}")
	if args.invert:
		im = 255 - im

	if args.pad and args.size_str.isdigit() and im.shape[0] != im.shape[1]:
			diff = abs(im.shape[0] - im.shape[1])
			if im.shape[0] > im.shape[1]:
				pad_arg = ( (0, 0), (diff / 2, diff / 2 + diff % 2) )
			else:
				pad_arg = ( (diff / 2, diff / 2 + diff % 2), (0, 0) )
			if len(im.shape) == 3:
				pad_arg += ( (0,0), ) 
			im = np.pad(im, pad_arg, mode='constant', constant_values=args.pad_val)

	if args.dense_surf:
		ims = extract_surf(im, args)
	else:
		size = get_size(im.shape[:2], args)
		ims = [skimage.transform.resize(im, size)]
	return ims

def get_surf_instance():
	if cv2.__version__.startswith('3'):
		surf = cv2.xfeatures2d.SURF_create()
		surf.setExtended(False)
		surf.setUpright(True)
	else:
		surf = cv2.SURF()
		surf.upright = True
		surf.extended = False
	return surf


def extract_surf(im, args):
	surf = get_surf_instance()
	spacing_y = im.shape[0] / args.surf_grid_size
	spacing_x = im.shape[1] / args.surf_grid_size
	ys = list(np.arange(args.surf_grid_size) * spacing_y)
	xs = list(np.arange(args.surf_grid_size) * spacing_x)
	radius = max(spacing_y, spacing_x) * args.surf_radius
	kps = [cv2.KeyPoint(x, y, radius, 0,) for y in ys for x in xs]
	kps, descriptors = surf.compute(im, kps)

	dense = descriptors.reshape( (args.surf_grid_size, args.surf_grid_size, 64) )
	dense = 128 * (dense + 1) 
	dense = dense.astype(np.uint8)
	channel_start = 0
	ims = list()
	while channel_start < 64:
		im_slice = np.squeeze(dense[:,:,channel_start:channel_start+3])
		ims.append(im_slice)
		channel_start += 3
	return ims
	


def open_db(db_file):
	env = lmdb.open(db_file, readonly=False, map_size=int(2 ** 42), max_readers=2000, writemap=True)
	txn = env.begin(write=True)
	return env, txn
	
def package(im, dbid, args):
	doc_datum = caffe.proto.caffe_pb2.DocumentDatum()
	doc_datum.dbid = dbid
	datum_im = doc_datum.image

	datum_im.channels = im.shape[2] if len(im.shape) == 3 else 1
	datum_im.width = im.shape[1]
	datum_im.height = im.shape[0]
	datum_im.encoding = args.encoding

	# image data
	if args.encoding != 'none':
		buf = StringIO.StringIO()
		if datum_im.channels == 1:
			plt.imsave(buf, im, format=args.encoding, vmin=0, vmax=1, cmap='gray')
		else:
			plt.imsave(buf, im, format=args.encoding, vmin=0, vmax=1)
		datum_im.data = buf.getvalue()
	else:
		pix = im.transpose(2, 0, 1)
		datum_im.data = pix.tostring()

	return doc_datum

def main(args):
	dbs = {}
	if args.multiple_db:
		try:
			os.makedirs(args.outdb)
		except:
			pass

	# delete and recreate
	if args.out_dir:
		try:
			if os.path.exists(args.out_dir):
				shutil.rmtree(args.out_dir)
		except:
			print traceback.print_exc(file=sys.stdout)
			print "Could not remove:", args.out_dir
		try:
			os.makedirs(args.out_dir)
		except:
			print traceback.print_exc(file=sys.stdout)
			print "Could not create:", args.out_dir


	print "Reading Manifest..."
	lines = open(args.manifest, 'r').readlines()
	if args.shuffle:
		print "Shuffling Data..."
		random.shuffle(lines)

	c = collections.Counter()
	db_c = collections.Counter()
	for x,line in enumerate(lines):
		if x and x % 1000 == 0:
			print "Processed %d images" % x
		try:
			line = line.rstrip()
			tokens = line.split()
			im_file = os.path.join(args.imroot, tokens[0])
			ims = process_im(im_file, args)
			dbid = int(tokens[1])

			for idx, im in enumerate(ims):
				doc_datum = package(im, dbid, args)
				if args.multiple_db:
					db_file = os.path.join(args.outdb, "%dx%d_lmdb" % im.shape[:2])
				elif args.dense_surf:
					db_file = os.path.join(args.outdb, "surf_%d_lmdb" % idx)
				else:
					db_file = args.outdb
				if db_file not in dbs:
					dbs[db_file] = open_db(db_file)
					print "\tOpening %s" % db_file
				env, txn = dbs[db_file]
				key = "%d:%s" % (x, os.path.splitext(os.path.basename(im_file))[0])
				txn.put(key, doc_datum.SerializeToString())
				c[db_file] += 1
				if c[db_file] % 1000 == 0:
					txn.commit()
					env.sync()
					print db_file
					print env.stat()
					print env.info()
					txn = env.begin(write=True)
					dbs[db_file] = (env, txn)

				if args.out_dir and dbid not in db_c:
					dirname = os.path.join(args.out_dir, str(dbid))
					try:
						os.makedirs(dirname)
					except:
						print traceback.print_exc(file=sys.stdout)
						print "Could not create:", dirname
						
				db_c[dbid] += 1
				if args.out_dir and db_c[dbid] <= args.num_samples:
					base = os.path.splitext(os.path.basename(im_file))[0]
					fname1 = os.path.join(args.out_dir, str(dbid), "%s.png" % base)
					fname2 = os.path.join(args.out_dir, str(dbid), "%s_db_entry.%s" % (base, args.encoding))
					try:
						skimage.io.imsave(fname1, im)
						open(fname2, 'wb').write(doc_datum.image.data)
					except:
						print traceback.print_exc(file=sys.stdout)
						print "Could not save to:", fname1
						print "Original image: ", im_file

		except Exception as e:
			print e
			print traceback.print_exc(file=sys.stdout)
			print "Error occured on:", im_file


	print "Done Processing Images"

	for key, val in dbs.items():
		print "Closing DB: ", key
		env, txn = val
		txn.commit()
		env.close()


def get_args():
	parser = argparse.ArgumentParser(description="Creates an LMDB of DocumentDatums")
	parser.add_argument('manifest', type=str,
						help='file listing image-paths and metadata-paths, one per line')
	parser.add_argument('imroot', type=str,
						help='path to the root of the image dir')
	parser.add_argument('outdb', type=str,
						help='where to put the db')

	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('-c', '--color', default=False, action="store_true",
						help='Force images to be RGB.')
	group.add_argument('-g', '--gray', default=False, action="store_true",
						help='Force images to be grayscale.')
	group.add_argument('-b', '--binary', default=False, action="store_true",
						help='Force images to be binarized at full resolution, then resized as grayscale.')
	group.add_argument('-d', '--dense-surf', default=False, action="store_true",
						help='output surf descriptors in a dense grid')

	parser.add_argument('-e', '--encoding', type=str, default='none',
						help='How to store the image in the DocumentDatum')
	parser.add_argument('--no-shuffle', dest="shuffle", default=True, action="store_false",
						help='How to store the image in the DocumentDatum')
	parser.add_argument('-s', '--size-str', type=str, default="",
						help='The size string: e.g. 256, 256x384, 256l, 384s')
	parser.add_argument('-i', '--invert', default=False, action="store_true",
						help='Whether to invert the images or not')
	parser.add_argument('-p', '--pad', default=False, action="store_true",
						help='Whether to pad images to preserve aspect ratio when using square size-str')
	parser.add_argument('--pad-val', default=0, type=int, 
						help='Value used in padding')
	parser.add_argument('--surf-grid-size', default=100, type=int, 
						help='Size of the surf grid')
	parser.add_argument('--surf-radius', default=2, type=float, 
						help='Size of each surf descriptor as a multiple of the grid resolution')

	group = parser.add_argument_group('Multiple AR', 'These parameters control the creation of many LMDBS, with one per Aspect Ratio')
	group.add_argument('-a', '--aspect-ratio-bin', type=int, default=32,
						help='When sizing image, round it to the nearest aspect ratio')
	group.add_argument('--aspect-ratio-bin2', type=float, default=0.15,
						help='When sizing image, round it to the nearest aspect ratio')
	group.add_argument('-m', '--multiple-db', default=False, action="store_true",
						help='Output a db for each image size in the $outdb directory')
	group.add_argument('-t', '--truncate', type=float, default='2',
						help='Upper/Lower bound on the image AR')

	group = parser.add_argument_group('Samples', 'Save samples of processed images')
	parser.add_argument('-o', '--out-dir', type=str, default='',
						help='Where to store samples of each class.  Will delete any existing directory.')
	parser.add_argument('-n', '--num-samples', type=int, default=1,
						help='How many samples of each class to save')
	
	args = parser.parse_args()
	return args



if __name__ == "__main__":
	args = get_args();
	print "%s started with..." % __file__
	print args
	main(args)

