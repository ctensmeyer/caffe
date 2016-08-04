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
import collections
import lmdb

ROOT="/fslgroup/fslg_nnml/compute"

SIZES=[512,384,256,150,100,64,32]

OUTPUT_SIZES = {"andoc_1m": 974, "andoc_1m_10": 974, "andoc_1m_50": 974, "rvl_cdip": 16, "rvl_cdip_10": 16, "rvl_cdip_100": 16, "imagenet": 1000}

MEAN_VALUES = { "andoc_1m": {"binary": [194], "binary_invert": [61], "gray": [175], "gray_invert": [80], "gray_padded": [None], "color": [178,175,166], "color_invert": [77,80,89], "color_padded": [126,124,118], "color_multiple": [178,175,166]},
				"rvl_cdip": {"binary": [233], "binary_invert": [22], "gray": [234], "gray_padded": [239], "gray_invert": [21], "gray_multiple": [234]},
				"imagenet": {"color": [104,117,123]},
			  }
MEAN_VALUES['rvl_cdip_10'] = MEAN_VALUES['rvl_cdip']
MEAN_VALUES['rvl_cdip_100'] = MEAN_VALUES['rvl_cdip']
MEAN_VALUES['andoc_1m_10'] = MEAN_VALUES['andoc_1m']
MEAN_VALUES['andoc_1m_50'] = MEAN_VALUES['andoc_1m']

DEFAULT_TEST_TRANSFORMS = [10]

def lmdb_num_entries(db_path):
	env = lmdb.open(db_path, readonly=True)
	stats = env.stat()
	num_entries = stats['entries']
	env.close()
	return num_entries
	

def OUTPUT_FOLDER(dataset, group, experiment, split):
	return os.path.join("experiments/preprocessing/nets" , dataset, group, experiment, split)

def OUTPUT_FOLDER_BINARIZE(dataset, group, experiment, split):
	return os.path.join("experiments/binarize/nets" , dataset, group, experiment, split)

def TRANSFORMS_FOLDER(dataset, group, experiment, split):
	return os.path.join(OUTPUT_FOLDER(dataset,group,experiment,split), "transforms")

def EXPERIMENTS_FOLDER(dataset, group, experiment, split):
	return os.path.join(ROOT, OUTPUT_FOLDER(dataset, group, experiment, split))

def EXPERIMENTS_FOLDER_BINARIZE(dataset, group, experiment, split):
	return os.path.join(ROOT, OUTPUT_FOLDER_BINARIZE(dataset, group, experiment, split))

def LMDB_MULTIPLE_PATH(dataset, tag, split):
	lmdbs = collections.defaultdict(list)
	for s in 'train_lmdb', 'val_lmdb', 'test_lmdb':
		par_dir = os.path.join(ROOT, "lmdb", dataset, tag, split, s)
		for s_dir in os.listdir(par_dir):
			r_dir = os.path.join(par_dir, s_dir)
			lmdbs[s].append(r_dir)
		
		if dataset == 'rvl_cdip_10' or dataset == 'rvl_cdip_100':
			dataset = 'rvl_cdip'
		if dataset == 'andoc_1m_10' or dataset == 'andoc_1m_50':
			dataset = 'andoc_1m'

	return lmdbs['train_lmdb'], lmdbs['val_lmdb'], lmdbs['test_lmdb']
	

def LMDB_PATH(dataset, tag, split):
	#return map(lambda s: os.path.join(ROOT, "lmdb", dataset, tag, split, s), ["train_lmdb", "val_lmdb", "test_lmdb"])
	lmdbs = list()
	lmdbs.append(os.path.join(ROOT, "lmdb", dataset, tag, split, "train_lmdb"))
	if dataset == 'rvl_cdip_10' or dataset == 'rvl_cdip_100':
		dataset = 'rvl_cdip'
	if dataset == 'andoc_1m_10' or dataset == 'andoc_1m_50':
		dataset = 'andoc_1m'
	for s in ['val_lmdb', 'test_lmdb']:
		lmdbs.append(os.path.join(ROOT, "lmdb", dataset, tag, split, s))
	return lmdbs

def LMDB_PATH_BINARIZE(dataset, tag, data_partition='train'):
	if not isinstance(tag, basestring):
		path = os.path.join(ROOT, "lmdb", dataset, tag[0], "%s_%s_lmdb" % (tag[1], data_partition))
	else:
		path = os.path.join(ROOT, "lmdb", dataset, tag, "%s_%s_lmdb" % (tag, data_partition))
	return path
	

def getSizeFromTag(t):
	tokens = t.split("_")
	return int(tokens[1])
	#return map(int, re.sub("(_?[^0-9_]+_?)","", t).split("_"))

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

def convLayerSigmoid(prev, **kwargs):
	conv = L.Convolution(prev, param=[dict(lr_mult=1), dict(lr_mult=2)], weight_filler=dict(type='msra'), **kwargs)
	sigmoid = L.Sigmoid(conv, in_place=True)
	return sigmoid

def convLayerOnly(prev, **kwargs):
	conv = L.Convolution(prev, param=[dict(lr_mult=1), dict(lr_mult=2)], weight_filler=dict(type='msra'), **kwargs)
	return conv

def ipLayer(prev, **kwargs):
   return L.InnerProduct(prev, param=[dict(lr_mult=1), dict(lr_mult=2)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant'), **kwargs) 

def fcLayer(prev, **kwargs):
	fc = ipLayer(prev, **kwargs)
	relu = L.ReLU(fc, in_place=True)
	return relu
	
# all sized for 227x227
DEPTH_LAYERS = { 0 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],
                 1 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],
                 2 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],
                 3 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],
                 4 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv6", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],
	
                 5 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv6", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv7", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

                 6 : [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv6", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv7", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv8", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					]
				}
				
CONV_LAYERS = {
			   32:  [(convLayer, {"name": "conv1", "kernel_size": 5, "num_output": 24, "stride": 1}), 
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":64, "pad": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":96, "pad": 1}),
					 (poolLayer, {"name": "pool3", "kernel_size": 3, "stride": 2}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":96, "pad": 0}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":64, "pad": 0}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   64:  [(convLayer, {"name": "conv1", "kernel_size": 7, "num_output": 32, "stride": 1}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":96, "pad": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":148, "pad": 1}),
					 (poolLayer, {"name": "pool3", "kernel_size": 3, "stride": 2}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":148, "pad": 0}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":96, "pad": 0}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   100: [(convLayer, {"name": "conv1", "kernel_size": 9, "num_output": 48, "stride": 2}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":128, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":192, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":192, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":128, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   150: [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 64, "stride": 3}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":192, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":192, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],


			   227: [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   256: [(convLayer, {"name": "conv1", "kernel_size": 11, "num_output": 96, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 0}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   320: [(convLayer, {"name": "conv1", "kernel_size": 15, "num_output": 96, "stride": 5}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 5, "num_output":256, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":384, "pad": 0}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":256, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   384: [(convLayer, {"name": "conv1", "kernel_size": 15, "num_output": 120, "stride": 3}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 7, "num_output":320, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 5, "num_output":448, "pad": 1}),
					 (poolLayer, {"name": "pool3", "kernel_size": 3, "stride": 2}),
					 (convLayer, {"name": "conv4", "kernel_size": 3, "num_output":448, "pad": 0}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":320, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],

			   512: [(convLayer, {"name": "conv1", "kernel_size": 15, "num_output": 144, "stride": 4}), 
					 (poolLayer, {"name": "pool1", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm1", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv2", "kernel_size": 7, "num_output":384, "pad": 2}),
					 (poolLayer, {"name": "pool2", "kernel_size": 3, "stride": 2}),
					 (L.LRN,	 {"name": "norm2", "local_size": 5, "alpha": 0.0001, "beta": 0.75}),
					 (convLayer, {"name": "conv3", "kernel_size": 5, "num_output":512, "pad": 1}),
					 (poolLayer, {"name": "pool3", "kernel_size": 3, "stride": 2}),
					 (convLayer, {"name": "conv4", "kernel_size": 5, "num_output":512, "pad": 1}),
					 (convLayer, {"name": "conv5", "kernel_size": 3, "num_output":384, "pad": 1}),
					 (poolLayer, {"name": "pool5", "kernel_size": 3, "stride": 2})
					],
			  }



FC_LAYERS = {

			 32:  [(fcLayer, {"name": "fc6", "num_output": 1024}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 1024}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 64:  [(fcLayer, {"name": "fc6", "num_output": 1536}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 1536}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 100: [(fcLayer, {"name": "fc6", "num_output": 2048}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 2048}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 150: [(fcLayer, {"name": "fc6", "num_output": 3072}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 3072}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 227: [(fcLayer, {"name": "fc6", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 256: [(fcLayer, {"name": "fc6", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 320: [(fcLayer, {"name": "fc6", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 4096}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],
				   
			 384: [(fcLayer, {"name": "fc6", "num_output": 5120}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 5120}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],

			 512: [(fcLayer, {"name": "fc6", "num_output": 6144}),
				   (L.Dropout, {"name": "dropout6", "dropout_ratio": 0.5, "in_place": True}),
				   (fcLayer, {"name": "fc7", "num_output": 6144}),
				   (L.Dropout, {"name": "dropout7", "dropout_ratio": 0.5, "in_place": True})],
				   
			}


VAL_BATCH_SIZE = 40
TRAIN_VAL = "train_val.prototxt"
TRAIN_TEST = "train_test.prototxt"
TRAIN_TEST2 = "train_test2.prototxt"
DEPLOY_FILE = "deploy.prototxt"
SOLVER = "solver.prototxt"
SNAPSHOT_FOLDER = "snapshots"

LEARNING_RATES = {"andoc_1m": 0.005, "rvl_cdip": 0.003, "imagenet": 0.01}
BATCH_SIZE = {"andoc_1m": 128, "rvl_cdip": 32, "imagenet": 256}
MAX_ITER = {"andoc_1m": 250000, "rvl_cdip": 500000, "imagenet": 450000}
STEP_SIZE = {"andoc_1m": 100000, "rvl_cdip": 150000, "imagenet": 100000}

for d in [LEARNING_RATES, BATCH_SIZE, MAX_ITER, STEP_SIZE]:
	d['rvl_cdip_10'] = d['rvl_cdip']
	d['rvl_cdip_100'] = d['rvl_cdip']
	d['andoc_1m_10'] = d['andoc_1m']
	d['andoc_1m_50'] = d['andoc_1m']

MAX_ITER['rvl_cdip_100'] = 150000
STEP_SIZE['rvl_cdip_100'] = 50000

SOLVER_PARAM = {#"test_iter": 1000, 
				"test_interval": 1000, 
				"lr_policy": '"step"',
				"gamma": 0.1,
				"display": 20,
				"momentum": 0.9,
				"weight_decay": 0.0005,
				"snapshot": 1000,
				"solver_mode": "GPU"}

# this is for validation sets, not test sets.  Test set iterations are specified in train.sh
MULTIPLE_TEST_ITERS  = { "andoc_1m" : { "color_227_multiple": 1007, "color_384_multiple": 1013, 
									    "color_227_multiple2": 1005, "color_384_multiple2": 1005},
						 "rvl_cdip" : { "gray_227_multiple":  1006, "gray_384_multiple": 1008, 
						 			    "gray_227_multiple2": 1005, "gray_384_multiple2": 1005}
                       }

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

def createElasticDeformationParam(elastic_sigma, elastic_max_alpha):
	return dict(sigma=elastic_sigma, max_alpha=elastic_max_alpha)


def createTransformParam2(scale, shift,  seed):
	params = []
	if scale is not None and shift is not None:
		params.append({'linear_params': {'scale': scale, 'shift': shift}})
	elif scale is not None:
		params.append({'linear_params': {'scale': scale, 'shift': 0}})
	elif shift is not None:
		params.append({'linear_params': {'scale': 1.0, 'shift': shift}})
	return {'params': params, 'rng_seed': seed}
		


def createTransformParam(phase, seed=None, test_transforms = DEFAULT_TEST_TRANSFORMS, deploy=False, **kwargs):
	params = []

	if deploy:
		tt = test_transforms
		transforms = {}
		for t in tt:
			transforms[t] = []
		if not kwargs.get('crop'):
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
				h = kwargs.get('hmirror', 0)
				v = kwargs.get('vmirror', 0)

				if 'shear' not in kwargs and 'crop' not in kwargs:
					for t in tt:
						transforms[t].extend(make_transforms.make_mirror_transforms(h,v))
						break


		#Perspective
		if 'perspective' in kwargs:
			params.append(dict(perspective_params = createPerspectiveParam(kwargs['perspective'])))
			
			if deploy:
				for t in tt:
					transforms[t].extend(make_transforms.make_perspective_transforms(kwargs['perspective'], t))

		#Elastic
		if 'elastic_sigma' in kwargs:
			params.append(dict(elastic_deformation_params = createElasticDeformationParam(kwargs['elastic_sigma'], kwargs['elastic_max_alpha'])))
			
			if deploy:
				for t in tt:
					transforms[t].extend(make_transforms.make_elastic_deformation_transforms(kwargs['elastic_sigma'], kwargs['elastic_max_alpha'], t))

		#rotate
		if 'rotation' in kwargs:
			params.append(dict(rotate_params = createRotateParam(kwargs['rotation'])))
			if deploy:
				for t in tt:
					transforms[t].extend(make_transforms.make_rotation_transforms(kwargs['rotation'], t))

		if 'shear' in kwargs:
			params.append(dict(shear_params = createShearParam(kwargs['shear']))) 
		
			if deploy and 'hmirror' not in kwargs and 'vmirror' not in kwargs and 'crop' not in kwargs:
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
	if kwargs.get('crop'):
		params.append(dict(crop_params = createCropParam(phase)))
 
		if deploy and 'hmirror' not in kwargs and 'vmirror' not in kwargs and 'shear' not in kwargs:
			for t in tt:
				im_size = kwargs['im_size']
				crop_size = kwargs['crop']
				transforms[t].extend(make_transforms.make_crop_transforms(im_size, crop_size, int(round(np.sqrt(t)))))

	# For combined data augmentation. This is pretty messy
	if deploy:
		h = kwargs.get('hmirror', 0)
		v = kwargs.get('vmirror', 0)
		im_size = kwargs.get('im_size', None)
		angle = kwargs.get('shear', None)
		repeats = kwargs.get('shear_repeats', 1)
		if 'crop' in kwargs and 'shear' in kwargs and ('hmirror' in kwargs or 'vmirror' in kwargs):
			transforms['crop_shear_mirror'] = make_transforms.make_crop_shear_mirror_transforms(im_size, 227, h, v, angle, repeats)

		if 'crop' in kwargs and 'shear' in kwargs:
			transforms['crop_shear'] = make_transforms.make_crop_shear_transforms(im_size, 227, angle, repeats)
			transforms['crop'] = make_transforms.make_crop_transforms(im_size, 227, 3)
			transforms['shear'] = make_transforms.make_shear_transforms(angle, 10) 

		if 'crop' in kwargs and ('hmirror' in kwargs or 'vmirror' in kwargs):
			transforms['crop_mirror'] = make_transforms.make_crop_mirror_transforms(im_size, 227, h, v)
			transforms['crop'] = make_transforms.make_crop_transforms(im_size, 227, 3)
			transforms['mirror'] = make_transforms.make_mirror_transforms(h, v)

		if 'shear' in kwargs and ('hmirror' in kwargs or 'vmirror' in kwargs):
			transforms['shear_mirror'] = make_transforms.make_shear_mirror_transforms( h, v, angle, repeats)
			transforms['mirror'] = make_transforms.make_mirror_transforms(h, v)
			transforms['shear'] = make_transforms.make_shear_transforms(angle, 10) 


	p = dict(params=params)

	if seed != None:
		p['rng_seed'] = seed

	if deploy and "transforms_folder" in kwargs:
		for t, trans in transforms.items():
			if len(trans) == 0:
				continue

			filename = os.path.join(kwargs["transforms_folder"], "transforms_%s.txt" % (t))
			#print trans
			with open(filename, "w") as f:
				f.write('\n'.join(trans))

	return p


def createBinarizeNetwork(train_input_sources=[], train_label_sources=[], val_input_sources=[], val_label_sources=[], depth=3,
						  kernel_size=3, num_filters=24, num_scales=1, wfm_loss=True, global_features=0, deploy=False, seed=None):
	assert deploy or len(train_input_sources) == len(val_input_sources)
	assert deploy or len(train_label_sources) == len(val_label_sources)
	assert deploy or train_input_sources
	if seed == None:
		seed = random.randint(0, 2147483647)
	n = caffe.NetSpec()	
	data_param = dict(backend=P.Data.LMDB)

	if deploy:
		n.data = L.Input()
	else:
		# training inputs
		inputs = list()

		if num_scales > 1:
			first_input, height, width = L.DocData(sources=[train_input_sources[0]], include=dict(phase=caffe.TRAIN), 
				ntop=3, label_names=['height', 'width'], backend=P.Data.LMDB, batch_size=1, 
				image_transform_param=createTransformParam2(scale=(1./255), shift=127, seed=seed))
			inputs.append(first_input)
			n.dims = L.Concat(height, width, include=dict(phase=caffe.TRAIN))
		else:
			first_input = L.DocData(sources=[train_input_sources[0]], include=dict(phase=caffe.TRAIN), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam2(scale=(1./255), shift=127, seed=seed))
			inputs.append(first_input)

		for source in train_input_sources[1:]:
			input = L.DocData(sources=[source], include=dict(phase=caffe.TRAIN), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam2(scale=(1./255), shift=127, seed=seed))
			inputs.append(input)

		if len(inputs) == 1:
			n.data = inputs[0]
		else:
			n.data = L.Concat(*inputs, include=dict(phase=caffe.TRAIN))

		# training labels
		n.gt = L.DocData(sources=[train_label_sources[0]], include=dict(phase=caffe.TRAIN), batch_size=1, backend=P.Data.LMDB)
			

		if wfm_loss:
			n.recall_weights = L.DocData(sources=[train_label_sources[1]], include=dict(phase=caffe.TRAIN), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam2(scale=(1./128), shift=None, seed=seed))
			n.precision_weights = L.DocData(sources=[train_label_sources[2]], include=dict(phase=caffe.TRAIN), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam2(scale=(1./128), shift=None, seed=seed))

		# val inputs
		inputs = list()

		if num_scales > 1:
			first_input, height, width = L.DocData(sources=[val_input_sources[0]], include=dict(phase=caffe.TEST), 
				ntop=3, label_names=['height', 'width'], batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam2(scale=(1./255), shift=127, seed=seed))
			inputs.append(first_input)
			n.VAL_dims = L.Concat(height, width, include=dict(phase=caffe.TEST))
		else:
			first_input = L.DocData(sources=[val_input_sources[0]], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam2(scale=(1./255), shift=127, seed=seed))
			inputs.append(first_input)

		for source in val_input_sources[1:]:
			input = L.DocData(sources=[source], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam2(scale=(1./255), shift=127, seed=seed))
			inputs.append(input)

		if len(inputs) == 1:
			n.VAL_data = inputs[0]
		else:
			n.VAL_data = L.Concat(*inputs, include=dict(phase=caffe.TRAIN))
			
		# val labels
		n.VAL_gt = L.DocData(sources=[val_label_sources[0]], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB)
			

		if wfm_loss:
			n.VAL_recall_weights = L.DocData(sources=[val_label_sources[1]], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam2(scale=(1./128), shift=None, seed=seed))
			n.VAL_precision_weights = L.DocData(sources=[val_label_sources[2]], include=dict(phase=caffe.TEST), batch_size=1, backend=P.Data.LMDB,
				image_transform_param=createTransformParam2(scale=(1./128), shift=None, seed=seed))

	# middle layers
	prev_layer = n.data
	pad_size = (kernel_size - 1) / 2
	layers = collections.defaultdict(list)
	for scale in xrange(num_scales):
		for conv_idx in xrange(depth - scale):  # is depth - scale the right thing to do?
			prev_layer = convLayer(prev_layer, kernel_size=kernel_size, pad=pad_size, num_output=num_filters, stride=1)
			layers[scale].append(prev_layer)
		if scale < (num_scales - 1):
			# not last scale
			prev_layer = poolLayer(layers[scale][0], kernel_size=2, stride=2)

	if global_features > 0:
		prev_layer = convLayer(layers[num_scales-1][0], kernel_size=3, pad=1, num_output=num_filters, stride=2)
		prev_layer = L.Pooling(prev_layer, pool=P.Pooling.AVE, global_pooling=True)
		for conv_idx in xrange(global_features): 
			prev_layer = convLayer(prev_layer, kernel_size=1, pad=0, num_output=num_filters, stride=1)
			layers[num_scales].append(prev_layer)

	last_layers = []
	for scale in xrange(num_scales):
		scale_layers = layers[scale]
		if scale_layers:
			last_layers.append(scale_layers[-1])
	
	if len(last_layers) > 1:
		# resize smaller scales to original size
		for idx in xrange(len(last_layers)):
			if idx == 0:
				continue
			last_layers[idx] = L.BilinearInterpolation(last_layers[idx], n.dims)
		n.merged = L.Merge(*last_layers)
		prev_layer = convLayer(n.merged, kernel_size=1, pad=0, num_output=num_filters, stride=1)

	# output/loss layer
	if wfm_loss:
		prev_layer = convLayerSigmoid(prev_layer, kernel_size=kernel_size, pad=pad_size, num_output=1, stride=1)
		if not deploy: 
			n.weighted_fmeasure = L.WeightedFmeasureLoss(prev_layer, n.gt, n.recall_weights, n.precision_weights)
			n.precision, n.recall, n.accuracy, n.nmr = L.PR(prev_layer, n.gt, ntop=4, include=dict(phase=caffe.TEST))
		else:
			output = prev_layer
	else:
		prev_layer = convLayerOnly(prev_layer, kernel_size=kernel_size, pad=pad_size, num_output=1, stride=1)
		if not deploy:
			n.class_loss = L.SigmoidCrossEntropyLoss(prev_layer, n.gt)
			prev_layer = L.Sigmoid(prev_layer, include=dict(phase=caffe.TEST))
			n.precision, n.recall, n.accuracy, n.nmr = L.PR(prev_layer, n.gt, ntop=4, include=dict(phase=caffe.TEST))
		else:
			output = L.Sigmoid(prev_layer)


	return n.to_proto()


def createNetwork(sources, size, val_sources=None,  num_output=1000, concat=False, pool=None, batch_size=32, deploy=False, 
					seed=None, shift_channels=None, scale_channels=None, multiple=False, val_batch_size=VAL_BATCH_SIZE, **tparams):
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
				val_first, val_targets = L.DocData(sources = [val_sources[0]], include=dict(phase=caffe.TEST), batch_size=val_batch_size,
						image_transform_param=createTransformParam(caffe.TEST, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
						label_names=["dbid"], ntop=2, **data_param)

				val_inputs = map(lambda s, t: L.DocData(sources=[s], include=dict(phase=caffe.TEST), batch_size=val_batch_size, 
					image_transform_param=createTransformParam(caffe.TEST, shift=t[0], scale=t[1], **tparams), **data_param), 
					val_sources[1:], zip(shift_channels[1:],scale_channels[1:]))
		
				n.VAL_data = L.Concat(val_first, *val_inputs, name="val_data", include=dict(phase=caffe.TEST))
				n.VAL_labels = val_targets
			
	
		else:
			data_param['ntop'] = 2
			data_param['label_names'] = ["dbid"]

			if multiple:
				# enable weight by size because we have multiple lmdbs to read from
				n.data, n.labels = L.DocData(sources = sources, batch_size=batch_size, include=dict(phase=caffe.TRAIN), 
					image_transform_param=createTransformParam(caffe.TRAIN, seed=seed, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
					weights_by_size=True, **data_param)
			else:
				n.data, n.labels = L.DocData(sources = sources, batch_size=batch_size, include=dict(phase=caffe.TRAIN), 
					image_transform_param=createTransformParam(caffe.TRAIN, seed=seed, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
					**data_param)


			if val_sources:
				if multiple:
					# make sure in_order and no_wrap are true so we can iterate the whole validation set through multiple lmdbs
					n.VAL_data, n.VAL_labels = L.DocData(sources=val_sources, name="validation", batch_size=val_batch_size, 
						include=dict(phase=caffe.TEST),  
						image_transform_param=createTransformParam(caffe.TEST, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
						in_order=True, no_wrap=True, **data_param) 
				else:
					n.VAL_data, n.VAL_labels = L.DocData(sources=val_sources, name="validation", batch_size=val_batch_size, 
						include=dict(phase=caffe.TEST), 
						image_transform_param=createTransformParam(caffe.TEST, shift=shift_channels[0], scale=scale_channels[0], **tparams), 
						**data_param) 
	else:
		createTransformParam(caffe.TEST, shift=shift_channels[0], scale=scale_channels[0], deploy=deploy, **tparams)
		n.data = L.Input()

	#CONV layers
	if 'depth' in tparams:
		layers = DEPTH_LAYERS[tparams['depth']]
	else:
		layers = CONV_LAYERS[size]
	layer = n.data
	for t, kwargs in layers[:-1]:
		if (tparams.get('width_mult') or tparams.get('conv_width_mult')) and kwargs.get('num_output'):
			kwargs = kwargs.copy()
			mult = tparams.get('width_mult')
			if not mult:
				mult = tparams.get('conv_width_mult')
			kwargs['num_output'] = int(mult * kwargs['num_output'])
		layer = t(layer, **kwargs)

	if pool is not None:
		#print "ADDING PADDING"
		# add in a padding to powers of 2 so that we guarentee the same size output
		layer = L.Padding(layer, pad_to_power_of_2=True, name="padding")
		if pool == 'spp':
			layer = L.SPP(layer, pyramid_height=4, name="spp")
		elif pool == 'hvp':
			layer = L.HVP(layer, num_horz_partitions=3, num_vert_partitions=3, name="hvp")
	else:
		layer = layers[-1][0](layer, **layers[-1][1])

			
	
	#FC layers
	fc_layers = FC_LAYERS[size]
	for t, kwargs in fc_layers:
		if (tparams.get('width_mult') or tparams.get('fc_width_mult')) and kwargs.get('num_output'):
			kwargs = kwargs.copy()
			mult = tparams.get('width_mult')
			if not mult:
				mult = tparams.get('fc_width_mult')
			kwargs['num_output'] = int(mult * kwargs['num_output'])
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

def createBinarizeExperiment(ds, tags, group, experiment, num_experiments=1, wfm_loss=True, lr=0.01, **kwargs):

	sources_input_train = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, data_partition='train'), tags)
	sources_input_val1 = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, data_partition='val1'), tags)
	sources_input_val2 = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, data_partition='val2'), tags)
	sources_input_test = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, data_partition='test'), tags)

	label_tags = ['processed_gt', 'recall_weights', 'precision_weights']
	sources_label_train = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, data_partition='train'), label_tags)
	sources_label_val1 = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, data_partition='val1'), label_tags)
	sources_label_val2 = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, data_partition='val2'), label_tags)
	sources_label_test = map(lambda tag: LMDB_PATH_BINARIZE(ds, tag, data_partition='test'), label_tags)

	params = {'train_input_sources': sources_input_train, 'train_label_sources': sources_label_train, 'wfm_loss': wfm_loss}
	params.update(kwargs)
	
	for exp_num in range(1,num_experiments+1):
		exp_num = str(exp_num)

		out_dir = OUTPUT_FOLDER_BINARIZE(ds, group, experiment, exp_num)
		print out_dir

		if not os.path.exists(out_dir):
			print "Directory Not Found, Creating"
			os.makedirs(out_dir)

		#create train_val file
		train_val = os.path.join(out_dir, TRAIN_VAL)
		with open(train_val, "w") as f:
			n = str(createBinarizeNetwork(val_input_sources=sources_input_val1, val_label_sources=sources_label_val1, **params))
			f.write(re.sub("VAL_", "", n))
	
		#Create train_test file
		train_test = os.path.join(out_dir, TRAIN_TEST)
		with open(train_test, "w") as f:
			n = str(createBinarizeNetwork(val_input_sources=sources_input_val2, val_label_sources=sources_label_val2, **params))
			f.write(re.sub("VAL_", "", n))

		train_test = os.path.join(out_dir, TRAIN_TEST2)
		with open(train_test, "w") as f:
			n = str(createBinarizeNetwork(val_input_sources=sources_input_test, val_label_sources=sources_label_test, **params))
			f.write(re.sub("VAL_", "", n))

		#Create Deploy File
		deploy_file = os.path.join(out_dir, DEPLOY_FILE)
		with open(deploy_file, "w") as f:
			n = createBinarizeNetwork(deploy=True, **params)
			for i, l in enumerate(n.layer):
				if l.type == "Input":
					del n.layer[i]
					break

			n.input.extend(['data'])
			n.input_dim.extend([1, 3 * len(sources_input_train),1024,1024])
			f.write(str(n))

		exp_folder = EXPERIMENTS_FOLDER_BINARIZE(ds,group,experiment,exp_num)
		snapshot_solver = os.path.join(exp_folder, SNAPSHOT_FOLDER, experiment)
		train_val_solver = os.path.join(exp_folder, TRAIN_VAL)


		solver = os.path.join(out_dir, SOLVER)
		with open(solver, "w") as f:
			f.write("net: \"%s\"\n" % (train_val_solver))
			f.write("base_lr: %f\n" % lr)
			f.write("gamma: %f\n" % 0.1)
			f.write("monitor_test: true\n")
			f.write("monitor_test_id: 0\n")
			f.write("max_steps_without_improvement: %d\n" % 5)
			f.write("max_periods_without_improvement: %d\n" % 6)
			f.write("min_lr: %f\n" % 1e-8)
			f.write("max_iter: %d\n" % 100000)

			f.write("test_iter: %d\n" % lmdb_num_entries(sources_label_val1[0]))
			f.write("test_interval: %d\n" % 100)
			f.write("snapshot: %d\n" % 100)
			f.write("momentum: %f\n" % 0.9)
			f.write("weight_decay: %f\n" % 0.0005)

			f.write("display: %d\n" % 1)
			f.write("solver_mode: GPU\n")
			f.write("snapshot_prefix: \"%s\"" % (snapshot_solver))
	

def createExperiment(ds, tags, group, experiment, num_experiments=1, pool=None, multiple=False, shift=None, scale=None, **tparams):

	# Check if tags are all the same size or not
	# If they aren't we are doing multi-scale training, and need to stick them all
	# in the same doc data layer 
	# TODO: Pyramid input
	if not isinstance(tags, list):
		tags = [tags]
	sizes = map(getSizeFromTag, tags)
	size = sizes[0]
	same_size = (not multiple)  # multiple AR training defaults to not the same size
	for s in sizes:
		same_size = (same_size and s == size)

	im_size = size
	tags_noSize = map(getTagWithoutSize, tags)
	if shift == "mean":
		shift = map(lambda t: MEAN_VALUES[ds][t], tags_noSize)

	if tparams.get('crop'):
		same_size = True
		size = tparams['crop']
	
	#if sizes are different, spatial pyramid pooling is required.
	if not same_size and pool is None:
		raise Exception("Input DBs are not the same size and regular pooling is enabled")

	for exp_num in range(1,num_experiments+1):
		exp_num = str(exp_num)

		out_dir = OUTPUT_FOLDER(ds, group, experiment, exp_num)
		print out_dir

		if not os.path.exists(out_dir):
			print "Directory Not Found, Creating"
			os.makedirs(out_dir)
		tf = TRANSFORMS_FOLDER(ds, group, experiment, exp_num)
		if not os.path.exists(tf):
			os.makedirs(tf)
		
		if multiple:
			sources_tr, sources_val, sources_ts = [], [], []
			for tag in tags:
				tr, val, ts = LMDB_MULTIPLE_PATH(ds, tag, "1")
				sources_tr += tr
				sources_val += val
				sources_ts += ts
		else:
			# only 1 lmdb split is in current use
			sources = map(lambda t: LMDB_PATH(ds, t, "1"), tags)
			sources_tr, sources_val, sources_ts =  zip(*sources)

		#common parameters
		params = dict(sources=list(sources_tr), size=size, num_output=OUTPUT_SIZES[ds], concat=same_size, 
					pool=pool, shift_channels=shift, scale_channels=scale, batch_size=BATCH_SIZE[ds], 
					multiple=multiple, **tparams)
		params['val_batch_size'] = VAL_BATCH_SIZE if ds != 'imagenet' else 50
	   
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
			n.input_dim.extend([1,getNumChannels(tags_noSize),size,size])
			f.write(str(n))

		#Create snapshot directory
		snapshot_out = os.path.join(out_dir,SNAPSHOT_FOLDER)
		if not os.path.exists(snapshot_out):
			print "Snapshot Directory Not Found, Creating"
			os.makedirs(snapshot_out)


		exp_folder = EXPERIMENTS_FOLDER(ds,group,experiment,exp_num)
		snapshot_solver = os.path.join(exp_folder, SNAPSHOT_FOLDER, experiment)
		train_val_solver = os.path.join(exp_folder, TRAIN_VAL)

		solver = os.path.join(out_dir, SOLVER)
		with open(solver, "w") as f:
			f.write("net: \"%s\"\n" % (train_val_solver))
			f.write("base_lr: %f\n" % (LEARNING_RATES[ds]))
			f.write("max_iter: %d\n" % (MAX_ITER[ds]))
			f.write("stepsize: %d\n" % (STEP_SIZE[ds]))
			if multiple:
				f.write("test_iter: %d\n" % MULTIPLE_TEST_ITERS[ds][tag]) 
			else:
				f.write("test_iter: %d\n" % 1000)  # for all non-multiple datasets
			for param, val in SOLVER_PARAM.items():
				f.write("%s: %s\n" % (param, str(val)))

			f.write("snapshot_prefix: \"%s\"" % (snapshot_solver))
		

