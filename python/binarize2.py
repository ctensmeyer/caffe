#!/usr/bin/python

import os
import sys
import argparse
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import caffe
import cv2
import scipy.ndimage.morphology



LEFT_EDGE = -2
TOP_EDGE = -1
MIDDLE = 0
RIGHT_EDGE = 1
BOTTOM_EDGE = 2

def safe_mkdir(_dir):
	try:
		os.makedirs(_dir)
	except:
		pass


def setup_network(args):
	net_file = os.path.join(args.net_dir, 'deploy.prototxt')
	weights_file = os.path.join(args.net_dir, 'weights.caffemodel')
	network = caffe.Net(net_file, weights_file, caffe.TEST)
	if args.gpu >= 0:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()
		
	return network


def fprop(network, ims, args):
	idx = 0
	responses = list()
	while idx < len(ims):
		sub_ims = ims[idx:idx+args.batch_size]
		network.blobs["data"].reshape(len(sub_ims), ims[0].shape[2], ims[0].shape[0], ims[0].shape[1]) 
		for x, im in enumerate(sub_ims):
			transposed = np.transpose(im, [2,0,1])
			transposed = transposed[np.newaxis, :, :, :]
			network.blobs["data"].data[x,:,:,:] = transposed
		idx += args.batch_size

		# propagate on batch
		network.forward()
		output = np.copy(network.blobs["prob"].data)
		responses.append(output)
		print "Progress %d%%" % int(100 * min(idx, len(ims)) / float(len(ims)))
	return np.concatenate(responses, axis=0)


def predict(network, ims, args):
	all_outputs = np.squeeze(fprop(network, ims, args), axis=1)

	high_indices = all_outputs >= args.threshold
	predictions = np.zeros_like(all_outputs)
	predictions[high_indices] = 1
	return all_outputs, predictions


def get_subwindows(im, pad_size, tile_size):
	height, width = tile_size, tile_size
	y_stride, x_stride = tile_size - (2 * pad_size), tile_size - (2 * pad_size)
	if (height > im.shape[0]) or (width > im.shape[1]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, (height, width))
		exit(1)
	ims = list()
	locations = list()
	y = 0
	y_done = False
	while y  <= im.shape[0] and not y_done:
		x = 0
		if y + height > im.shape[0]:
			y = im.shape[0] - height
			y_done = True
		x_done = False
		while x <= im.shape[1] and not x_done:
			if x + width > im.shape[1]:
				x = im.shape[1] - width
				x_done = True
			locations.append( ((y, x, y + height, x + width), 
					(y + pad_size, x + pad_size, y + y_stride, x + x_stride),
					 TOP_EDGE if y == 0 else (BOTTOM_EDGE if y == (im.shape[0] - height) else MIDDLE),
					  LEFT_EDGE if x == 0 else (RIGHT_EDGE if x == (im.shape[1] - width) else MIDDLE) 
			) )
			ims.append(im[y:y+height,x:x+width])
			x += x_stride
		y += y_stride

	return locations, ims


def stich_together(locations, subwindows, size, dtype, pad_size, tile_size):
	output = np.zeros(size, dtype=dtype)
	for location, subwindow in zip(locations, subwindows):
		outer_bounding_box, inner_bounding_box, y_type, x_type = location
		y_paste, x_paste, y_cut, x_cut, height_paste, width_paste = -1, -1, -1, -1, -1, -1

		if y_type == TOP_EDGE:
			y_cut = 0
			y_paste = 0
			height_paste = tile_size - pad_size
		elif y_type == MIDDLE:
			y_cut = pad_size
			y_paste = inner_bounding_box[0]
			height_paste = tile_size - 2 * pad_size
		elif y_type == BOTTOM_EDGE:
			y_cut = pad_size
			y_paste = inner_bounding_box[0]
			height_paste = tile_size - pad_size

		if x_type == LEFT_EDGE:
			x_cut = 0
			x_paste = 0
			width_paste = tile_size - pad_size
		elif x_type == MIDDLE:
			x_cut = pad_size
			x_paste = inner_bounding_box[1]
			width_paste = tile_size - 2 * pad_size
		elif x_type == RIGHT_EDGE:
			x_cut = pad_size
			x_paste = inner_bounding_box[1]
			width_paste = tile_size - pad_size

		output[y_paste:y_paste+height_paste, x_paste:x_paste+width_paste] = subwindow[y_cut:y_cut+height_paste, x_cut:x_cut+width_paste]

	return output


def save_histo(data, fname, title, weights=None):
	if weights is not None:
		weights = weights.flatten()
		
	n, bins, patches = plt.hist(data.flatten(), bins=100, weights=weights, log=True)
	plt.title(title)
	plt.xlabel('Predicted Probability of Foreground')
	plt.ylabel('Pixel Count')
	plt.tick_params(axis='y', which='minor', left='off', right='off')

	plt.savefig(fname)
	plt.clf()


def xor_image(im1, im2):
	out_image = np.zeros(im1.shape + (3,))

	for y in xrange(im1.shape[0]):
	    for x in xrange(im1.shape[1]):
			if im1[y,x]:
				if im2[y,x]:
					# white on white
					out_image[y,x] = (255,255,255)
				else:
					# white on black
					out_image[y,x] = (255,0,0)
			else:
				if im2[y,x]:
					# black on white
					out_image[y,x] = (0,255,0)
				else:
					# black on black
					out_image[y,x] = (0,0,0)
	return out_image


def get_ims_files(args):
	im_files = map(lambda s: s.strip(), open(args.image_manifest, 'r').readlines())
	im_dirs = map(lambda s: s.strip(), open(os.path.join(args.net_dir, 'inputs.txt'), 'r').readlines())
	return im_files, im_dirs


def get_pad_size(args):
	return int(open(os.path.join(args.net_dir, 'pad.txt'), 'r').read())


def load_im(im_file, im_dirs, args):
	ims = list()
	for im_dir in im_dirs:
		im_path = os.path.join(args.dataset_dir, im_dir, im_file)
		im = cv2.imread(im_path, -1)  # reads in image as is
		if im is None:
			raise Exception("File does not exist: %s" % im_path)
		if im.ndim == 2:
			im = im[:,:,np.newaxis]
		ims.append(im)
	im = np.concatenate(ims, axis=2)
	im = im - args.mean
	im = args.scale * im
	return im


def write_output(locations, raw_subwindows, binarized_subwindows, im_file, image, pad_size, im, args):
	binary_result = stich_together(locations, binarized_subwindows, tuple(image.shape[0:2]), 
		np.uint8, pad_size, args.tile_size)
	binary_result = 255 * (1 - binary_result)
	binary_out_file = os.path.join(args.out_dir, 'basic', im_file)
	cv2.imwrite(binary_out_file, binary_result)

	if args.verbose:
		out_dir = os.path.join(args.out_dir, 'verbose', os.path.splitext(os.path.basename(im_file))[0])
		safe_mkdir(out_dir)
		out_prefix = os.path.join(out_dir, os.path.splitext(im_file)[0])

		binary_out_file = out_prefix + "_pred.png"
		cv2.imwrite(binary_out_file, binary_result)

		# raw probabilities
		raw_result = stich_together(locations, raw_subwindows, tuple(image.shape[0:2]), 
			np.float, pad_size, args.tile_size)
		raw_out_file = out_prefix + "_raw.png"
		cv2.imwrite(raw_out_file, 255 * (1 - raw_result))

		# add in the gt file
		gt_file = os.path.join(args.dataset_dir, 'original_gt', im_file)
		gt_im = cv2.imread(gt_file, -1) / 255.
		gt_out_file = out_prefix + "_gt.png"
		cv2.imwrite(gt_out_file, 255 * gt_im)

		# difference between predicted and gt
		diff_im = xor_image(binary_result / 255, gt_im)
		diff_out_file = out_prefix + "_diff.png"
		cv2.imwrite(diff_out_file, diff_im)

		# histogram of probabilities
		save_histo(raw_result, out_prefix + "_histo_global.png", "Global")
		save_histo(raw_result, out_prefix + "_histo_background.png", "Background Pixels", gt_im)

		for x in xrange(3):
			# histogram of probabilites within x pixels of foreground
			save_histo(raw_result, out_prefix + "_histo_foreground_%d.png" % x, "Foreground Pixels (within %d)" % x, 1 - gt_im)
			gt_im = scipy.ndimage.morphology.binary_erosion(gt_im)

		if im.shape[2] == 1 or im.shape[2] == 3:
			cv2.imwrite(out_prefix + "_original_image.png", im / args.scale + args.mean)


def main(args):
	network = setup_network(args)
	im_files, im_dirs = get_ims_files(args)
	pad_size = get_pad_size(args)
	print "Pad Size:", pad_size
	safe_mkdir(os.path.join(args.out_dir, 'basic'))
	for idx, im_file in enumerate(im_files):
		image = load_im(im_file, im_dirs, args)
		cv2.imwrite('tmp.png', (1./ args.scale) * image + args.mean)
		locations, subwindows = get_subwindows(image, pad_size, args.tile_size)
		raw_subwindows, binarized_subwindows = predict(network, subwindows, args)
		write_output(locations, raw_subwindows, binarized_subwindows, im_file, image, pad_size, image, args)

		if idx and idx % args.print_count == 0:
			print "Processed %d/%d Images" % (idx, len(im_files))
	

def get_args():
	parser = argparse.ArgumentParser(description="Outputs binary predictions")

	parser.add_argument("net_dir", 
				help="The dir containing the model")
	parser.add_argument("dataset_dir",
				help="The dataset to be evaluated")
	parser.add_argument("image_manifest",
				help="txt file listing images to evaluate")
	parser.add_argument("out_dir",
				help="output directory")

	parser.add_argument("--gpu", type=int, default=0,
				help="GPU to use for running the network")

	parser.add_argument("-m", "--mean", type=float, default=127.,
				help="Mean value for data preprocessing")
	parser.add_argument("-a", "--scale", type=float, default=0.0039,
				help="Optional scale factor")
	parser.add_argument("--print-count", default=1, type=int, 
				help="Print every print-count images processed")
	parser.add_argument("-b", "--batch-size", default=1, type=int, 
				help="Max number of transforms in single batch per original image")
	parser.add_argument("-t", "--tile-size", default=128, type=int, 
				help="Size of tiles to extract")
	parser.add_argument("--threshold", default=0.5, type=float, 
				help="Probability threshold for foreground/background")
	parser.add_argument("-v", "--verbose", default=False, action='store_true',
				help="Write auxiliary images for analysis")

	args = parser.parse_args()

	return args
			

if __name__ == "__main__":
	args = get_args()
	main(args)

