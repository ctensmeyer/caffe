#!/usr/bin/python

import argparse
import os
import re

import make_transforms

import caffe
from caffe import layers as L
from caffe import params as P

import numpy as np
import random

DEBUG=False
ROOT="/fslgroup/fslg_nnml/compute"

DATASET_NAMES=["andoc_1m", "rvl_cdip"]
DATASET_TAGS={"andoc_1m": ["binary_227", "color_150", "color_256", "color_384", "color_64",
						   "binary_227_invert", "color_227", "color_256_padded", "color_384_padded", 
						   "gray_227", "color_100", "color_227_invert", "color_32", "color_512", "gray_227_invert"],
			  
			  "rvl_cdip": ["binary_227", "gray_150", "gray_256", "gray_384", "gray_64",
						   "binary_227_invert", "gray_227", "gray_256_padded", "gray_384_padded", 
						   "gray_100", "gray_227_invert", "gray_32", "gray_512"]
			  } 

SPLITS=["1","2","3"]
SIZES=[512,384,256,150,100,64,32]
EXPERIMENT_GROUPS = ["augmentation", "resolution", "channel", "padding"]

OUTPUT_SIZES = {"andoc_1m": 974, "rvl_cdip": 16}

MEAN_VALUES = { "andoc_1m": {"binary": [194], "binary_invert": [61], "gray": [175], "gray_invert": [80], "gray_padded": [None], "color": [178,175,166], "color_invert": [77,80,89], "color_padded": [126,124,118]},
				"rvl_cdip": {"binary": [233], "binary_invert": [22], "gray": [234], "gray_padded": [180], "gray_invert": [21]}
			  }


def OUTPUT_FOLDER(dataset, group, experiment, split):
	if DEBUG:
		return os.path.join("debug" , dataset, group, experiment, split)
	else:
		return os.path.join("experiments/preprocessing/nets" , dataset, group, experiment, split)


def TRANSFORMS_FOLDER(dataset, group, experiment, split):
	return os.path.join(OUTPUT_FOLDER(dataset,group,experiment,split), "transforms")

def EXPERIMENTS_FOLDER(dataset, group, experiment, split):
	return os.path.join(ROOT, OUTPUT_FOLDER(dataset, group, experiment, split))

def LMDB_PATH(dataset, tag, split):
	return map(lambda s: os.path.join(ROOT, "lmdb", dataset, tag, split, s), ["train_lmdb", "val_lmdb", "test_lmdb"])

def getSizeFromTag(t):
	return map(int, re.sub("(_?[^0-9_]+_?)","", t).split("_"))

def getTagWithoutSize(t):
	return re.sub("_*[0-9]+","", t)

def getNumChannels(tags):
	channels = 0

	for t in tags:
		if "color" in t:
			channels += 3
		elif "gray" in t:
			channels += 1
		elif "binary" in t:
			channels += 1

	return channels


def poolLayer(prev, **kwargs):
	return L.Pooling(prev, pool=P.Pooling.MAX, **kwargs)

def convLayer(prev, **kwargs):
	conv = L.Convolution(prev, param=[dict(lr_mult=1), dict(lr_mult=2)], weight_filler=dict(type='msra'), **kwargs)
	relu = L.ReLU(conv, in_place=True)
	return relu

def ipLayer(prev, **kwargs):
   return L.InnerProduct(prev, param=[dict(lr_mult=1), dict(lr_mult=2)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant'), **kwargs) 

def fcLayer(prev, **kwargs):
	fc = ipLayer(prev, **kwargs)
	relu = L.ReLU(fc, in_place=True)
	return relu
	
CONV_LAYERS = {227: [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					]
					 
			  }



FC_LAYERS = {227: [(fcLayer, {"name": "fc6", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})]}


VAL_BATCH_SIZE = 40
TRAIN_VAL = "train_val.prototxt"
TRAIN_TEST = "train_test.prototxt"
DEPLOY_FILE = "deploy.prototxt"
SOLVER = "solver.prototxt"
SNAPSHOT_FOLDER = "snapshots"

LEARNING_RATES = {"andoc_1m": 0.005, "rvl_cdip": 0.003}
BATCH_SIZE = {"andoc_1m": 128, "rvl_cdip": 32}
MAX_ITER = {"andoc_1m": 250000, "rvl_cdip": 650000}
STEP_SIZE = {"andoc_1m": 100000, "rvl_cdip": 150000}

SOLVER_PARAM = {"test_iter": 1000, 
				"test_interval": 1000, 
				"lr_policy": '"step"',
				"gamma": 0.1,
				"display": 20,
				"momentum": 0.9,
				"weight_decay": 0.0005,
				"snapshot": 1000,
				"solver_mode": "GPU"}

def createLinearParam(shift=0.0, scale=1.0, **kwargs):
	return dict(shift=shift, scale=scale)

def createColorJitterParam(sigma=5):
	return dict(sigma=sigma)

def createCropParam(phase):
	if phase == caffe.TRAIN:
		location = P.CropTransform.RANDOM
	else:
		location = P.CropTransform.CENTER

	return dict(size=227, location=location)

def createReflectParam(hmirror=0.0, vmirror=0.0, **kwargs):
	p = {}
	if hmirror != None:
		p['horz_reflect_prob'] = hmirror
	
	if vmirror != None:
		p['vert_reflect_prob'] = vmirror

	return p
	

def createNoiseParam(low, high=None):
	std = [low]

	if high != None:
		std.append(high)

	return dict(std_dev=std)


def createRotateParam(rotation):
	return dict(max_angle=rotation)

def createShearParam(shear):
	return dict(max_shear_angle=shear)

def createBlurParam(blur):
	return dict(max_sigma = blur)

def createUnsharpParam(params):
	if isinstance(params, dict):
		return params
	else:
		return dict(max_sigma=params)

def createPerspectiveParam(sig):
	return dict(max_sigma=sig)


def createTransformParam(phase, seed=None, test_transforms = [10,50,100], deploy=False, **kwargs):
	params = []

	if deploy:
		tt = test_transforms
		transforms = {}
		for t in tt:
			transforms[t] = []
		transforms[1] = ['none']

	#noise
	if (phase == caffe.TRAIN or deploy) and 'noise_std' in kwargs:
		noise = kwargs['noise_std']

		if not isinstance(noise, list):
			noise = [noise]

		params.append(dict(gauss_noise_params = createNoiseParam(*noise)))

		if deploy:
			for t in tt:
				transforms[t].extend(make_transforms.make_gaussnoise_transforms(noise[1], t))

	# color jitter
	if (phase == caffe.TRAIN or deploy) and 'color_std' in kwargs:
		sigma = kwargs['color_std']

		params.append(dict(color_jitter_params = createColorJitterParam(sigma)))

		if deploy:
			for t in tt:
				transforms[t].extend(make_transforms.make_color_jitter_transforms(sigma, t))

	#linear
	if 'scale' in kwargs or 'shift' in kwargs:
		params.append(dict(linear_params = createLinearParam(**kwargs)))

   
	if phase == caffe.TRAIN or deploy:
		#mirror
		if 'hmirror' in kwargs or 'vmirror' in kwargs:
			params.append(dict(reflect_params = createReflectParam(**kwargs)))
			if deploy:
				if 'hmirror' in kwargs:
					h = kwargs['hmirror']
				else:
					h = 0

				if 'vmirror' in kwargs:
					v = kwargs['vmirror']
				else:
					v = 0

				for t in tt:
					transforms[t].extend(make_transforms.make_mirror_transforms(h,v))
					break


		#Perspective
		if 'perspective' in kwargs:
			params.append(dict(perspective_params = createPerspectiveParam(kwargs['perspective'])))
			
			if deploy:
				for t in tt:
					transforms[t].extend(make_transforms.make_perspective_transforms(kwargs['perspective'], t))

		#rotate
		if 'rotation' in kwargs:
			params.append(dict(rotate_params = createRotateParam(kwargs['rotation'])))
			if deploy:
				for t in tt:
					transforms[t].extend(make_transforms.make_rotation_transforms(kwargs['rotation'], t))

		if 'shear' in kwargs:
			params.append(dict(shear_params = createShearParam(kwargs['shear']))) 
		
			if deploy:
				for t in tt:
					transforms[t].extend(make_transforms.make_shear_transforms(kwargs['shear'], t))


		#blur
		p = {}
		if 'blur' in kwargs:
			p['gauss_blur_params'] = createBlurParam(kwargs['blur'])
		
			if deploy:
				split = 2 if 'unsharp' in kwargs else 1
				for t in tt:
					transforms[t].extend(make_transforms.make_blur_transforms(kwargs['blur'], t/split))


		#unsharp
		if 'unsharp' in kwargs:
			p['unsharp_mask_params'] = createUnsharpParam(kwargs['unsharp'])

			if deploy:
				split = 2 if 'blur' in kwargs else 1
				for t in tt:
					transforms[t].extend(make_transforms.make_unsharp_transforms(kwargs['unsharp'], t))


		if len(p) > 0:
			params.append(p)

	#crop
	if 'crop' in kwargs and kwargs['crop']:
		params.append(dict(crop_params = createCropParam(phase)))
 
		if deploy:
			for t in tt:
				im_size = kwargs['im_size']
				transforms[t].extend(make_transforms.make_crop_transforms(im_size, 227, int(round(np.sqrt(t)))))

	p = dict(params=params)

	if seed != None:
		p['rng_seed'] = seed

	if deploy and "transforms_folder" in kwargs:
		for t, trans in transforms.items():
			if len(trans) == 0:
				continue

			filename = os.path.join(kwargs["transforms_folder"], "transforms_%d.txt" % (t))
			#print trans
			with open(filename, "w") as f:
				f.write('\n'.join(trans))

	return p




#Assume caffe net for a moment
def createNetwork(sources, size, val_sources=None,  num_output=1000, concat=False, spp=False, batch_size=32, deploy=False, 
					seed=None, shift_channels=None, scale_channels=None, **tparams):
	n = caffe.NetSpec()	
	#data
	data_param = dict(backend=P.Data.LMDB)

	if len(sources) == 1:
		concat = False
	
	
	#Helper function for checking transform params
	def checkTransform(trans, default):
		#If trans is not defined, replace with default
		if not trans:
			trans = default

		#If Shift channels is only one value, 
		if (not isinstance(trans, list)):
			trans = [trans]*len(sources)

		return trans

	#if shift_channels != None:
	shift_channels = checkTransform(shift_channels, 0)
	
	#if scale_channels != None:
	scale_channels = checkTransform(scale_channels, 1.0)   

	if seed == None:
		seed = random.randint(0, 2147483647)
	
	if not deploy:
		if concat:
			first, targets = L.DocData(sources = [sources[0]], include=dict(phase=caffe.TRAIN), batch_size=batch_size, 
					image_transform_param=createTransformParam(caffe.TRAIN, seed=seed, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
					label_names=["dbid"], ntop=2, **data_param)

			#print len(sources)
			inputs = map(lambda s, t: L.DocData(sources=[s], include=dict(phase=caffe.TRAIN), batch_size=batch_size, 
				image_transform_param=createTransformParam(caffe.TRAIN, seed=seed, shift=t[0], scale=t[1], **tparams) ,**data_param), sources[1:], 
				zip(shift_channels[1:], scale_channels[1:]))
		
			#print inputs
			n.data = L.Concat(first, *inputs, include=dict(phase=caffe.TRAIN))
			n.labels = targets
		
			if val_sources:
				val_first, val_targets = L.DocData(sources = [val_sources[0]], include=dict(phase=caffe.TEST), batch_size=VAL_BATCH_SIZE,
						image_transform_param=createTransformParam(caffe.TEST, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
						label_names=["dbid"], ntop=2, **data_param)

				val_inputs = map(lambda s, t: L.DocData(sources=[s], include=dict(phase=caffe.TEST), batch_size=VAL_BATCH_SIZE, 
					image_transform_param=createTransformParam(caffe.TEST, shift=t[0], scale=t[1], **tparams), **data_param), 
					val_sources[1:], zip(shift_channels[1:],scale_channels[1:]))
		
				n.VAL_data = L.Concat(val_first, *val_inputs, name="val_data", include=dict(phase=caffe.TEST))
				n.VAL_labels = val_targets
			
	
		else:
			data_param['ntop'] = 2
			data_param['label_names'] = ["dbid"]

			n.data, n.labels = L.DocData(sources = sources, batch_size=batch_size, include=dict(phase=caffe.TRAIN), 
					image_transform_param=createTransformParam(caffe.TRAIN, seed=seed, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
					**data_param)

			if val_sources:
				n.VAL_data, n.VAL_labels = L.DocData(sources=val_sources, name="validation", batch_size=VAL_BATCH_SIZE, 
						include=dict(phase=caffe.TEST), 
						image_transform_param=createTransformParam(caffe.TEST, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
						**data_param) 
	else:
		createTransformParam(caffe.TEST, shift=shift_channels[0], scale=scale_channels[0], deploy=deploy, **tparams)
		n.data = L.Input()

	#CONV layers
	layers = CONV_LAYERS[size]
	layer = n.data
	for t, kwargs in layers[:-1]:
		layer = t(layer, **kwargs)

	#If SPP and if last layer is pooling dont add pooling layer
	if not (spp and layers[-1][0] == poolLayer):
		layer = layers[-1][0](layer, **layers[-1][1])

	#add SPP
	if spp:
		#print "ADDING SPP"
		layer = L.SPP(layer,pyramid_height=4, name="spp")
	
	#FC layers
	fc_layers = FC_LAYERS[size]
	for t, kwargs in fc_layers:
		layer = t(layer, **kwargs)

	#Output Layer
	top = ipLayer(layer, name="top", num_output=num_output)

	n.top = top

	if not deploy:
		n.loss = L.SoftmaxWithLoss(n.top, n.labels)

		n.accuracy = L.Accuracy(n.top, n.labels)
	else:
		n.prob = L.Softmax(n.top)
	
	return n.to_proto()


def createExperiment(ds, tags, group, experiment, num_experiments=1, spp = False, shift=None, scale=None, **tparams):
	#Determine if the tags are the same size
	#print tags
	
	if not isinstance(tags, list):
		tags = [tags]

	sizes = map(getSizeFromTag, tags)
	tags_noSize = map(getTagWithoutSize, tags)

	
	if shift == "mean":
		shift = map(lambda t: MEAN_VALUES[ds][t], tags_noSize)

	size = min(sizes)
	same = True

	for s in sizes:
		same = same and s == size

	im_size = size[0]

	if 'crop' in tparams and tparams['crop'] == True:
		same = True
		size = [227]
	
	#if sizes are different, spatial pyramid pooling is required.
	if not same:
		spp = True

	for s in range(1,num_experiments+1):
		s = str(s)

		out_dir = OUTPUT_FOLDER(ds, group, experiment, s)
		print out_dir

		if not os.path.exists(out_dir):
			print "Directory Not Found, Creating"
			os.makedirs(out_dir)
		tf = TRANSFORMS_FOLDER(ds, group, experiment, s)
		if not os.path.exists(tf):
			os.makedirs(tf)
		
		sources = map(lambda t: LMDB_PATH(ds, t, "1"), tags)
		sources_tr, sources_val, sources_ts =  zip(*sources)

		#common parameters
		params = dict(sources=list(sources_tr), size=size[0], num_output=OUTPUT_SIZES[ds], concat=same, 
					spp=spp, shift_channels=shift, scale_channels=scale, batch_size=BATCH_SIZE[ds], **tparams)
	   
		#create train_val file
		train_val = os.path.join(out_dir, TRAIN_VAL)
		with open(train_val, "w") as f:
			n = str(createNetwork(val_sources=list(sources_val), **params))
			f.write(re.sub("VAL_", "", n))
	
		#Create train_test file
		train_test = os.path.join(out_dir, TRAIN_TEST)
		with open(train_test, "w") as f:
			n = str(createNetwork(val_sources=list(sources_ts), **params))
			f.write(re.sub("VAL_", "", n))

		#Create Deploy File
		deploy_file = os.path.join(out_dir, DEPLOY_FILE)
		with open(deploy_file, "w") as f:
			n = createNetwork(deploy=True, transforms_folder=tf,im_size=im_size, **params)
			for i, l in enumerate(n.layer):
				if l.type == "Input":
					del n.layer[i]
					break

			n.input.extend(['data'])
			n.input_dim.extend([1,getNumChannels(tags_noSize),size[0],size[0]])
			f.write(str(n))

		#Create snapshot directory
		snapshot_out = os.path.join(out_dir,SNAPSHOT_FOLDER)
		if not os.path.exists(snapshot_out):
			print "Snapshot Directory Not Found, Creating"
			os.makedirs(snapshot_out)


		exp_folder = EXPERIMENTS_FOLDER(ds,group,experiment,s)
		snapshot_solver = os.path.join(exp_folder, SNAPSHOT_FOLDER, experiment)
		train_val_solver = os.path.join(exp_folder, TRAIN_VAL)

		solver = os.path.join(out_dir, SOLVER)
		with open(solver, "w") as f:
			f.write("net: \"%s\"\n" % (train_val_solver))
			f.write("base_lr: %f\n" % (LEARNING_RATES[ds]))
			f.write("max_iter: %d\n" % (MAX_ITER[ds]))
			f.write("stepsize: %d\n" % (STEP_SIZE[ds]))
			for param, val in SOLVER_PARAM.items():
				f.write("%s: %s\n" % (param, str(val)))

			f.write("snapshot_prefix: \"%s\"" % (snapshot_solver))
		

def main(args):
	createExperiment(args.data_set, args.data_tags, args.experiment_group, args.experiment_tag, spp=args.spp, shift=args.shift, scale=args.scale)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Create Network Files and Solver Files.")
	parser.add_argument("data_set", help="The dataset name", choices=DATASET_NAMES)
	parser.add_argument("experiment_group", help="The group of the experiment")
	parser.add_argument("experiment_tag", help="The tag of the experiment")
	parser.add_argument("-dt", "--data_tags", help="The tag of the dataset", nargs="+", required=True)
	parser.add_argument("-spp", help="Enable Spatial Pyramid Pooling", action="store_true")
	parser.add_argument("--shift", help="Value by which to shift the data", type=float)
	parser.add_argument("--scale", help="Value by which to scale the data", type=float)

	main(parser.parse_args())
