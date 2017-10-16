
import os
import sys
import caffe
import caffe.io
import argparse
import shutil
import numpy as np
from PIL import Image

#np.set_printoptions(precision=8, linewidth=200, edgeitems=50)

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
	else:
		out_size = (int(args.size_str), int(args.size_str))
	return out_size

def resize(im, args):
	new_size = get_size(im.size, args)
	return im.resize(new_size)

def init_caffe(args):
	if args.gpu >= 0:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()

	caffenet = caffe.Net(args.caffe_model, args.caffe_weights, caffe.TEST)
	transformer = caffe.io.Transformer({args.input: caffenet.blobs[args.input].data.shape}, resize=False)
	transformer.set_transpose(args.input, (2,0,1))
	if args.mean_file:
		transformer.set_mean(args.input, np.load(args.mean_file).mean(1).mean(1))
	else:
		transformer.set_mean(args.input, np.asarray([-1 * args.shift] * (1 if args.gray else 3)))
	transformer.set_input_scale(args.input, args.scale)
	print "scale: ", args.scale

	if not args.gray:
		transformer.set_channel_swap(args.input, (2,1,0))
	caffenet.transformer = transformer

	return caffenet

def print_arch(net):
	def prod(l):
		p = 1
		for x in l:
			p *= x
		return p
	print "Blobs:"
	for name, blob in net.blobs.items():
		print "\t%s: %s" % (name, blob.data.shape)
	print

	num_params = 0
	print "Parameters:"
	for name, lblob in net.params.items():
		num_param = sum(map(lambda blob: prod(blob.data.shape), lblob))
		print "\t%s: %s\t%d" % (name, "\t".join(map(lambda blob: str(blob.data.shape), lblob)), num_param)
		num_params += num_param
	print
	print "Num Parameters:", num_params
	print

	print "Inputs:"
	for name in net.inputs:
		print "\t%s" % name
	print

	print "Outputs:"
	for name in net.outputs:
		print "\t%s" % name
	print

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def format_data(data, padsize=1, padval=0, local_norm=False):
	if local_norm:
		data -= data.min(axis=0, keepdims=True)
		data /= data.max(axis=0, keepdims=True)
	else:
		data -= data.min()
		data /= data.max()

	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
	data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
	# tile the filters into an image
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
	return data

_original = False
def format_general_filters(data, padsize=1, padval=0, local_norm=False):
	num_filters = data.shape[0]
	num_channels = data.shape[1]
	if _original:
		return format_data(data[:num_channels].reshape( (num_channels ** 2,) + data.shape[2:]), 
				local_norm=local_norm)
	
	if local_norm:
		data -= data.min(axis=0, keepdims=True)
		data /= data.max(axis=0, keepdims=True)
	else:
		data -= data.min()
		data /= data.max()
	padding = ( (0, 0), (0, 0), (0, padsize), (0, padsize) )
	data = np.pad(data, padding, mode='constant', constant_values=padval)
	
	data = data.reshape((num_filters, num_channels) + data.shape[2:]).transpose((0, 2, 1, 3))
	data = data.reshape((num_filters * data.shape[1], num_channels * data.shape[3]))
	return data
	

def save_blob(blob, out_file, args):
	if len(blob.shape) == 1:
		blob = np.reshape(blob, (1, blob.shape[0]))
	if len(blob.shape) == 4:
		num_filters = blob.shape[0]
		num_channels = blob.shape[1]
		width = blob.shape[2]
		height = blob.shape[3]

		if num_channels == 3:
			# RGB
			np_im = format_data(blob.transpose(0, 2, 3, 1), local_norm=args.local_norm)
		else:
			# this works if the number of channels is <= number of filters
			np_im = format_general_filters(blob, local_norm=args.local_norm)
	elif len(blob.shape) == 3:
		num_channels = blob.shape[0]
		if num_channels == 3:
			blob = np.reshape(blob, (1,) + blob.shape)
			np_im = format_data(blob.transpose(0, 2, 3, 1), local_norm=args.local_norm)
		else:
			np_im = format_data(blob, padval=1, local_norm=args.local_norm)
	else:
		np_im = blob - blob.min()
		np_im /= np_im.max()
	im = Image.fromarray(((255 * np_im).astype("uint8")))
	im = im.resize( (2 * im.size[0], 2 * im.size[1]), resample=Image.NEAREST)
	im.save(out_file)
	

def save_filters(net, args):
	out_dir = os.path.join(args.output_dir, "filters")
	os.mkdir(out_dir)
	for name, lblob in net.params.items():
		if name in args.omit_layers:
			print "Omitting %s" % name
			continue
		print name
		out_file = os.path.join(out_dir, name + ".png")
		weights = lblob[0].data
		save_blob(weights, out_file, args)

def save_activations(net, args):
	for im_f in args.test_images:
		print im_f
		outdir = os.path.join(args.output_dir, os.path.splitext(os.path.basename(im_f))[0])
		os.mkdir(outdir)
		im = Image.open(im_f)
		im = resize(im, args)

		if args.gray:
			im = im.convert("L")
		else:
			im = im.convert(mode='RGB')

		arr = np.array(im.getdata()).reshape(im.size[1], im.size[0], 1 if args.gray else 3)

		net.blobs[args.input].reshape(1, 1 if args.gray else 3, im.size[1], im.size[0])
		tmp = net.transformer.preprocess(args.input, arr)
		print tmp
		net.blobs[args.input].data[...] = tmp
		net.forward()
		for name, data in net.blobs.items():
			if name in args.omit_layers:
				print "Ommitting %s" % name
				continue
			try:
				print "\t%s: %s" % (name, data.data.shape)
				out_file = os.path.join(outdir, name + ".png")
				if len(data.data.shape) > 1:
					save_blob(data.data[0], out_file, args)
			except Exception as e:
				print "skipping %s" % name

def main(args):
	print "Initializing Network"
	net = init_caffe(args)
	print "Architecture"
	print_arch(net)
	if os.path.exists(args.output_dir):
		shutil.rmtree(args.output_dir)
	os.makedirs(args.output_dir)

	print "\nSaving Activations"
	save_activations(net, args)

	print "\nSaving Filters"
	save_filters(net, args)

def get_args():
	parser = argparse.ArgumentParser(description="Output network activations and filters")
	parser.add_argument('caffe_model', 
		help="prototxt file containing the network definition")
	parser.add_argument('caffe_weights', 
		help="prototxt file containing the network weights for the given definition")
	parser.add_argument('output_dir', 
		help="path to the output directory where images are written")

	parser.add_argument("-m", "--mean-file", type=str, default="",
				help="Optional mean file for input normalization")
	parser.add_argument("-s", "--size-str", type=str, default="227",
				help="Resize images to this size before processing")
	parser.add_argument("--gpu", type=int, default=-1,
				help="GPU to use for running the network")
	parser.add_argument('-g', '--gray', default=False, action="store_true",
						help='Force images to be grayscale.  Force color if ommited')
	parser.add_argument("-a", "--scale", type=float, default=1.0,
				help="Optional scale factor")
	parser.add_argument("-b", "--shift", type=float, default=0.0,
				help="Optional shift factor")
	parser.add_argument("-i", "--input", type=str, default="data",
				help="Name of input blob")
	parser.add_argument("-l", "--local_norm", default=False, action="store_true",
				help="Name of input blob")
	parser.add_argument("--omit", default="", type=str,
				help="Name of blobs to omit.  Common separated list")

	parser.add_argument('test_images', nargs=argparse.REMAINDER,
		help="images to run through the network")
	args = parser.parse_args()
	args.omit_layers = args.omit.split(',')

	return args

if __name__ == "__main__":
	args = get_args()
	main(args)

