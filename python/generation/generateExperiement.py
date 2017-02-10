import os
import createNetwork

DS = ["rvl_cdip", "andoc_1m", "rvl_cdip_10", "rvl_cdip_100", "andoc_1m_10", "andoc_1m_50", "imagenet"]

########################
#ds = 'rvl_cdip'
ds = 'andoc_1m'
#ds = 'combined'
#ds = 'imagenet'
#######################


def TAGS(T, size=227, pad=False, multiple=False, multiple2=False):
	t = T.lower()
	if t == 'b':
		tag = 'binary'
	elif t == 'g':
		tag = 'gray'
	elif t == 'c':
		tag = 'color'

	tag += "_%d" % (size)

	if T.isupper():
		tag += "_invert"

	if pad:
		tag += "_padded"

	if multiple:
		tag += "_multiple"

	if multiple2:
		tag += "_multiple2"

	return tag


def generateTag(T, size=227):
	return map(lambda t: TAGS(t, size), T)


def COMBO(ds=ds, size = 227, pad=False, multiple=False, multiple2=False):
	#print multiple
	if ds.startswith(DS[0]):
		#tags = ['g', 'b', 'G', 'B']
		#tags = ['g', 'G']
		tags = ['g']
	else:
		#tags = ['c', 'g', 'b', 'C', 'G', 'B']
		#tags = ['c', 'C']
		tags = ['c']

	return map(lambda t: TAGS(t, size, pad=pad, multiple=multiple, multiple2=multiple2), tags)


SIZES = [32, 64, 100, 150, 227, 256, 320, 384, 512]
MULTISCALE_SIZES = [(150, 227, 256), (227, 256, 320), (256, 320, 384), (320, 384, 512), (227, 256, 320, 384, 512)]
WIDTHS = [0.1, 0.25, 0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 2]
DEPTHS = [0, 1, 2, 3, 4, 5, 6]
DSURF_SIZES = [64, 100, 150, 227]
default = dict(shift="mean", scale=(1.0/255))
width_default = dict(shift="mean", scale=(1.0/255), shear=10, num_experiments=2)
size_default = dict(shift="mean", scale=(1.0/255), shear=10, num_experiments=2)
pad_default = dict(shift="mean", scale=(1.0/255), shear=10, num_experiments=2)
pad2_default = dict(shift="mean", scale=(1.0/255), crop_perc=0.9, num_experiments=2)

EXPERIMENTS = {"baseline": {"baseline" : (COMBO(ds), dict(num_experiments=3, **default))},
			   "standard": {"h_mirror" : (COMBO(ds), dict(hmirror=0.5, **default)),
							"v_mirror" : (COMBO(ds), dict(vmirror=0.5, **default)),
							"hv_mirror" : (COMBO(ds), dict(hmirror=0.5, vmirror=0.5, **default)),
				   
							"gauss_noise_5" : (COMBO(ds), dict(noise_std=[0,5], **default)),
							"gauss_noise_10" : (COMBO(ds), dict(noise_std=[0,10], **default)),
							"gauss_noise_15" : (COMBO(ds), dict(noise_std=[0,15], **default)),
							"gauss_noise_20" : (COMBO(ds), dict(noise_std=[0,20], **default)),
				   
							"crop_240" : (COMBO(ds,240), dict(crop=227, **default)),
							"crop_256" : (COMBO(ds,256), dict(crop=227, **default)),
							"crop_288" : (COMBO(ds,288), dict(crop=227, **default)),
							"crop_320" : (COMBO(ds,320), dict(crop=227, **default))
							},
				
				"rotate":  {"rotation_5":  (COMBO(ds), dict(rotation=5,  **default)),
							"rotation_10": (COMBO(ds), dict(rotation=10, **default)),
							"rotation_15": (COMBO(ds), dict(rotation=15, **default)),
							"rotation_20": (COMBO(ds), dict(rotation=20, **default)),
							},

				"shear":   {"shear_5":  (COMBO(ds), dict(shear=5,  **default)),
							"shear_10": (COMBO(ds), dict(shear=10, **default)),
							"shear_15": (COMBO(ds), dict(shear=15, **default)),
							"shear_20": (COMBO(ds), dict(shear=20, **default)),
							}, 
				
				"blur_sharp": {"gauss_blur_1_5": (COMBO(ds), dict(blur=1.5, **default)),
							   "gauss_blur_3":   (COMBO(ds), dict(blur=3, **default)),
							   
							   #"sharp_1_5":   (COMBO(ds), dict(unsharp=1.5, **default)),
							   #"sharp_3":   (COMBO(ds), dict(unsharp=3, **default)),
							   
							   #"sharp_blur_1_5":   (COMBO(ds), dict(blur=1.5, unsharp=1.5, **default)),
							   #"sharp_blur_3":   (COMBO(ds), dict(blur=3, unsharp=3, **default))
							   },

				"perspective": {"perspective_1": (COMBO(ds), dict(perspective=0.0001, **default)),
								"perspective_2": (COMBO(ds), dict(perspective=0.0002, **default)),
								"perspective_3": (COMBO(ds), dict(perspective=0.0003, **default)),
								"perspective_4": (COMBO(ds), dict(perspective=0.0004, **default))
								},
				"color_jitter":{"color_jitter_5": (COMBO(ds), dict(color_std=5, **default)),
								"color_jitter_10": (COMBO(ds), dict(color_std=10, **default)),
								"color_jitter_15": (COMBO(ds), dict(color_std=15, **default)),
								"color_jitter_20": (COMBO(ds), dict(color_std=20, **default)),
								},
				"elastic": { "elastic_2_5": (COMBO(ds), dict(elastic_sigma=2, elastic_max_alpha=5, **default)),
							 "elastic_2_10": (COMBO(ds), dict(elastic_sigma=2, elastic_max_alpha=10, **default)),
							 "elastic_3_5": (COMBO(ds), dict(elastic_sigma=3, elastic_max_alpha=5, **default)),
							 "elastic_3_10": (COMBO(ds), dict(elastic_sigma=3, elastic_max_alpha=10, **default)),
						   },
				"salt": {    "salt_025": (COMBO(ds), dict(salt_max_flip=0.025, **default)),
							 "salt_05": (COMBO(ds), dict(salt_max_flip=0.05, **default)),
							 "salt_1": (COMBO(ds), dict(salt_max_flip=0.1, **default)),
							 "salt_2": (COMBO(ds), dict(salt_max_flip=0.2, **default)),
						   },
				"combined": { "crop_mirror_shear": (COMBO(ds, 256), dict(crop=227, hmirror=0.0, vmirror=0.5, shear=10, num_experiments=2, shift="mean", scale=(1.0/255))),
							  "crop_mirror": (COMBO(ds, 256), dict(crop=227, hmirror=0.0, vmirror=0.5, num_experiments=2, shift="mean", scale=(1.0/255))),
							  "mirror_shear": (COMBO(ds), dict(hmirror=0.0, vmirror=0.5, shear=10, num_experiments=2, shift="mean", scale=(1.0/255))),
							  "crop_shear": (COMBO(ds, 256), dict(crop=227, shear=10, num_experiments=2, shift="mean", scale=(1.0/255))),
						   },

				"size": { "size_%d" % size: (COMBO(ds, size), dict(**size_default)) for size in SIZES },
				"size2": { "size_512_75": (COMBO(ds, 512), dict(width=0.75, **size_default)),
							 "size_512_50": (COMBO(ds, 512), dict(width=0.5, **size_default)) },
				"multiscale": { "multiscale_%d_%d" % (sizes[0], sizes[-1]): (COMBO(ds, sizes[-1]), 
					dict(pool='spp', sizes=sizes,**size_default)) for sizes in MULTISCALE_SIZES },

				"width": { "width_%d" % int(width * 100): (COMBO(ds), dict(width_mult=width, **width_default)) for width in WIDTHS },
				"width_2": { "width_conv_50" : (COMBO(ds), dict(conv_width_mult=0.50, **width_default)),
							 "width_conv_75" : (COMBO(ds), dict(conv_width_mult=0.75, **width_default)),
							 "width_fc_50" : (COMBO(ds), dict(fc_width_mult=0.50, **width_default)),
							 "width_fc_75" : (COMBO(ds), dict(fc_width_mult=0.75, **width_default)),
						   },
				"width3": { "width_conv_150" : (COMBO(ds), dict(conv_width_mult=1.50, **width_default)),
							 "width_conv_200" : (COMBO(ds), dict(conv_width_mult=2.00, **width_default)),
							 "width_conv_150_fc_50" : (COMBO(ds), dict(conv_width_mult=1.50, fc_width_mult=0.5, **width_default)),
						   },
				"padding2": {
							 "crop_227": (['color_227_short'], dict(num_experiments=2, crop=227, shear=10, **default)),
							 "crop_384": (['color_384_short'], dict(num_experiments=2, crop=384, shear=10, **default))
							},
				"padding4": {
							 "warped_227": (COMBO(ds, 227), dict(**pad_default)),
							 "warped_384": (COMBO(ds, 384), dict(**pad_default)),
							 "spp_warped_227": (COMBO(ds, 227), dict(pool='spp', **pad_default)),
							 "spp_warped_384": (COMBO(ds, 384), dict(pool='spp', **pad_default)),
							 "crop_227": (['color_227_short'], dict(num_experiments=2, crop=227, shear=10, **default)),
							 "crop_384": (['color_384_short'], dict(num_experiments=2, crop=384, shear=10, **default))
							},
				"padding": { 
							 "warped_227": (COMBO(ds, 227), dict(**pad_default)),
							 "pad_227":	(COMBO(ds, 227, pad=True), dict(**pad_default)),
							 "warped_384": (COMBO(ds, 384), dict(**pad_default)),
							 "pad_384":	(COMBO(ds, 384, pad=True), dict(**pad_default)),

							 "spp_warped_227": (COMBO(ds, 227), dict(pool='spp', **pad_default)),
							 "spp_pad_227":	(COMBO(ds, 227, pad=True), dict(pool='spp', **pad_default)),
							 "spp_warped_384": (COMBO(ds, 384), dict(pool='spp', **pad_default)),
							 "spp_pad_384":	(COMBO(ds, 384, pad=True), dict(pool='spp', **pad_default)),

							 "hvp_warped_227": (COMBO(ds, 227), dict(pool='hvp', **pad_default)),
							 "hvp_pad_227":	(COMBO(ds, 227, pad=True), dict(pool='hvp', **pad_default)),
							 "hvp_warped_384": (COMBO(ds, 384), dict(pool='hvp', **pad_default)),
							 "hvp_pad_384":	(COMBO(ds, 384, pad=True), dict(pool='hvp', **pad_default)),
						   },
				"multiple": {
							 "spp_multiple_227": (COMBO(ds, 227, multiple=True), dict(pool='spp', multiple=True, **pad_default)),
							 "spp_multiple_384": (COMBO(ds, 384, multiple=True), dict(pool='spp', multiple=True, **pad_default)),
							 "hvp_multiple_227": (COMBO(ds, 227, multiple=True), dict(pool='hvp', multiple=True, **pad_default)),
							 "hvp_multiple_384": (COMBO(ds, 384, multiple=True), dict(pool='hvp', multiple=True, **pad_default)),
							},
				"multiple2": {
							 "spp_multiple2_227": (COMBO(ds, 227, multiple2=True), dict(pool='spp', multiple=True, **pad_default)),
							 "spp_multiple2_384": (COMBO(ds, 384, multiple2=True), dict(pool='spp', multiple=True, **pad_default)),
							 "hvp_multiple2_227": (COMBO(ds, 227, multiple2=True), dict(pool='hvp', multiple=True, **pad_default)),
							 "hvp_multiple2_384": (COMBO(ds, 384, multiple2=True), dict(pool='hvp', multiple=True, **pad_default)),
							},
				"multiple2_crop": {
							 "spp_multiple2_227_crop": (COMBO(ds, 227, multiple2=True), dict(pool='spp', multiple=True, **pad2_default)),
							 "spp_multiple2_384_crop": (COMBO(ds, 384, multiple2=True), dict(pool='spp', multiple=True, **pad2_default)),
							 "hvp_multiple2_227_crop": (COMBO(ds, 227, multiple2=True), dict(pool='hvp', multiple=True, **pad2_default)),
							 "hvp_multiple2_384_crop": (COMBO(ds, 384, multiple2=True), dict(pool='hvp', multiple=True, **pad2_default)),
							},
				"depth": { "depth_%d" % depth: (COMBO(ds), dict(depth=depth, **width_default)) for depth in DEPTHS },
				"bn": { "depth_%d" % depth: (COMBO(ds), dict(depth=depth, **width_default)) for depth in [2,3,4,5] },
				"bn2": { "bn_dropout": (COMBO(ds, 227), dict(shear=10, num_experiments=2, bn=True, dropout=True, **default)),
						 "bn": (COMBO(ds, 227), dict(shear=10, num_experiments=2, bn=True, dropout=False, **default)),
						 "dropout": (COMBO(ds, 227), dict(shear=10, num_experiments=2, bn=False, dropout=True, **default)),
						 "neither": (COMBO(ds, 227), dict(shear=10, num_experiments=2, bn=False, dropout=False, **default)),
						 "no_lrn": (COMBO(ds, 227), dict(shear=10, num_experiments=2, lrn=False, **default))
					   },
				"dsurf": { "dsurf_%d" % size: (['dsurf_%d' % size], dict(dsurf=True, **width_default)) for size in DSURF_SIZES},
				"dsurf_gray": { "dsurf_gray_%d" % size: (['dsurf_%d' % size, 'gray_%d' % size], dict(dsurf=True, **width_default)) for size in DSURF_SIZES},

				"dsurf2": { "dsurf_227": (['dsurf_227'], dict(dsurf=True, num_experiments=10, **default))},
				"dsurf_gray2": { "dsurf_gray_227": (['dsurf_227', 'gray_227'], dict(dsurf=True, num_experiments=10, **default)) },
				"dsurf_color2": { "dsurf_color_227": (['dsurf_227', 'color_227'], dict(dsurf=True, num_experiments=10, **default)) },


				"finetune": { 
							  "depth_2_color": (['color_227'], dict(depth=2, conv_width_mult=1.5, **width_default)),
							  "depth_2_gray": (['gray_227'], dict(depth=2, conv_width_mult=1.5, **width_default)),
							  "depth_3_color": (['color_227'], dict(depth=3, conv_width_mult=1.5, **width_default)),
							  "depth_3_gray": (['gray_227'], dict(depth=3, conv_width_mult=1.5, **width_default)),
							  "depth_4_color": (['color_227'], dict(depth=4, conv_width_mult=1.5, **width_default)),
							  "depth_4_gray": (['gray_227'], dict(depth=4, conv_width_mult=1.5, **width_default)),
							 },
				"finetune2": { 
							  "andoc_depth_2_color": (['color_227'], dict(depth=2, conv_width_mult=1.5, **width_default)),
							  "andoc_depth_2_gray": (['gray_227'], dict(depth=2, conv_width_mult=1.5, **width_default)),
							  "andoc_depth_3_color": (['color_227'], dict(depth=3, conv_width_mult=1.5, **width_default)),
							  "andoc_depth_3_gray": (['gray_227'], dict(depth=3, conv_width_mult=1.5, **width_default)),
							  "andoc_depth_4_color": (['color_227'], dict(depth=4, conv_width_mult=1.5, **width_default)),
							  "andoc_depth_4_gray": (['gray_227'], dict(depth=4, conv_width_mult=1.5, **width_default)),
							  "imagenet_depth_2_color": (['color_227'], dict(depth=2, conv_width_mult=1.5, **width_default)),
							  "imagenet_depth_2_gray": (['gray_227'], dict(depth=2, conv_width_mult=1.5, **width_default)),
							  "imagenet_depth_3_color": (['color_227'], dict(depth=3, conv_width_mult=1.5, **width_default)),
							  "imagenet_depth_3_gray": (['gray_227'], dict(depth=3, conv_width_mult=1.5, **width_default)),
							  "imagenet_depth_4_color": (['color_227'], dict(depth=4, conv_width_mult=1.5, **width_default)),
							  "imagenet_depth_4_gray": (['gray_227'], dict(depth=4, conv_width_mult=1.5, **width_default)),
							  "combined_depth_2_color": (['color_227'], dict(depth=2, conv_width_mult=1.5, **width_default)),
							  "combined_depth_2_gray": (['gray_227'], dict(depth=2, conv_width_mult=1.5, **width_default)),
							  "combined_depth_3_color": (['color_227'], dict(depth=3, conv_width_mult=1.5, **width_default)),
							  "combined_depth_3_gray": (['gray_227'], dict(depth=3, conv_width_mult=1.5, **width_default)),
							  "combined_depth_4_color": (['color_227'], dict(depth=4, conv_width_mult=1.5, **width_default)),
							  "combined_depth_4_gray": (['gray_227'], dict(depth=4, conv_width_mult=1.5, **width_default)),
							  "cdip_depth_2_color": (['color_227'], dict(depth=2, conv_width_mult=1.5, **width_default)),
							  "cdip_depth_2_gray": (['gray_227'], dict(depth=2, conv_width_mult=1.5, **width_default)),
							  "cdip_depth_3_color": (['color_227'], dict(depth=3, conv_width_mult=1.5, **width_default)),
							  "cdip_depth_3_gray": (['gray_227'], dict(depth=3, conv_width_mult=1.5, **width_default)),
							  "cdip_depth_4_color": (['color_227'], dict(depth=4, conv_width_mult=1.5, **width_default)),
							  "cdip_depth_4_gray": (['gray_227'], dict(depth=4, conv_width_mult=1.5, **width_default)),
							 },
				"finetune3": { 
							  "color": (['color_227'], dict(shear=10, num_experiments=2, **default)),
							  "gray": (['gray_227'], dict(shear=10, num_experiments=2, **default)),
							 },
				"finetune4": { 
							  "andoc_color": (['color_227'], dict(shear=10, num_experiments=2, **default)),
							  "andoc_gray": (['gray_227'], dict(shear=10, num_experiments=2, **default)),
							  "cdip_color": (['color_227'], dict(shear=10, num_experiments=2, **default)),
							  "cdip_gray": (['gray_227'], dict(shear=10, num_experiments=2, **default)),
							  "imagenet_color": (['color_227'], dict(shear=10, num_experiments=2, **default)),
							  "imagenet_gray": (['gray_227'], dict(shear=10, num_experiments=2, **default)),
							  "combined_color": (['color_227'], dict(shear=10, num_experiments=2, **default)),
							  "combined_gray": (['gray_227'], dict(shear=10, num_experiments=2, **default)),
							 },
				"equi_arch": {
							  "conv_4": (COMBO(ds, 227), dict(depth=2, rotation=20, **default)),
							  "conv_5": (COMBO(ds, 227), dict(depth=3, rotation=20, **default)),
							  "conv_6": (COMBO(ds, 227), dict(depth=4, rotation=20, **default)),
							  "conv_7": (COMBO(ds, 227), dict(depth=5, rotation=20, **default)),
							  "fc_1": (COMBO(ds, 227), dict(fc_depth=1, rotation=20, **default)),
							  "fc_2": (COMBO(ds, 227), dict(fc_depth=2, rotation=20, **default)),
							  "fc_3": (COMBO(ds, 227), dict(fc_depth=3, rotation=20, **default)),
							  "fc_4": (COMBO(ds, 227), dict(fc_depth=4, rotation=20, **default)),
							  "width_50": (COMBO(ds, 227), dict(width_mult=0.5, rotation=20, **default)),
							  "width_150": (COMBO(ds, 227), dict(width_mult=1.5, rotation=20, **default)),
							 }
				}


def equiArchExperiments():
	group = "equi_arch"

	experiments = EXPERIMENTS["equi_arch"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)


def sizeExperiments():
	group = "size"

	experiments = EXPERIMENTS["size"]
	experiments.update(EXPERIMENTS["multiscale"])

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)

def size2Experiments():
	group = "size2"

	experiments = EXPERIMENTS["size2"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)

def depthExperiments():
	group = "depth"

	experiments = EXPERIMENTS["depth"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)


def paddingExperiments():
	group = "padding"

	experiments = EXPERIMENTS["padding"]
	#experiments = EXPERIMENTS["multiple2"]
	experiments.update(EXPERIMENTS["multiple2"])
	experiments.update(EXPERIMENTS["multiple2_crop"])

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)


def padding2Experiments():
	group = "padding3"

	experiments = EXPERIMENTS["padding2"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)


def padding4Experiments():
	group = "padding4"

	experiments = EXPERIMENTS["padding4"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)

def widthExperiments():
	group = "width"

	experiments = EXPERIMENTS["width"]
	experiments.update(EXPERIMENTS["width_2"])

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)

def width3Experiments():
	group = "width3"

	experiments = EXPERIMENTS["width3"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)


def finetuneExperiments():
	group = "finetune"

	experiments = EXPERIMENTS["finetune"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)

def finetune2Experiments():
	group = "finetune2"

	experiments = EXPERIMENTS["finetune2"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)

def finetune3Experiments():
	group = "finetune3"

	experiments = EXPERIMENTS["finetune3"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)

def finetune4Experiments():
	group = "finetune4"

	experiments = EXPERIMENTS["finetune4"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)


def bnExperiments():
	group = "bn"

	experiments = EXPERIMENTS["bn"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, bn=True, **tr)


def bn2Experiments():
	group = "bn2"

	experiments = EXPERIMENTS["bn2"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)


def dsurfExperiments():
	group = "dsurf"

	experiments = EXPERIMENTS["dsurf"]
	experiments.update(EXPERIMENTS["dsurf_gray"])

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)



def dsurf2Experiments():
	group = "dsurf2"

	experiments = EXPERIMENTS["dsurf2"]
	#experiments.update(EXPERIMENTS["dsurf_gray2"])
	experiments.update(EXPERIMENTS["dsurf_color2"])

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)


def augmentation2Experiments():
	group = "augmentation"
	
	experiments = EXPERIMENTS['salt']
	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)


def augmentation3Experiments():
	group = "aug_combined"
	
	experiments = EXPERIMENTS['combined']
	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)

def augmentationExperiments():
	group = "augmentation"
	
	#experiments = EXPERIMENTS["standard"]
	experiments = EXPERIMENTS['baseline']
	experiments.update(EXPERIMENTS["standard"])
	experiments.update(EXPERIMENTS["shear"])
	experiments.update(EXPERIMENTS["blur_sharp"])
	experiments.update(EXPERIMENTS["rotate"])
	experiments.update(EXPERIMENTS["shear"])
	experiments.update(EXPERIMENTS["perspective"])
	experiments.update(EXPERIMENTS["color_jitter"])
	experiments.update(EXPERIMENTS["elastic"])
	#experiments = EXPERIMENTS["combined"]

	for name, (tags, tr) in experiments.items():
		print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
		createNetwork.createExperiment(ds, tags, group, name, **tr)

def hsvExperiments():
	tr = {'num_experiments': 10, 'hsv': True, 'scale': (1./255), 'shift': 'mean'} 
	createNetwork.createExperiment(ds, ['color_227'], 'channel2', 'hsv', **tr)
	createNetwork.createExperiment(ds, ['color_227', 'color_227'], 'channel2', 'hsv_rgb', **tr)
	

def channel2Experiments():
	tr = {'num_experiments': 10, 'scale': (1./255), 'shift': 'mean'} 
	createNetwork.createExperiment(ds, ['binary_227'], 'channel2', 'binary', **tr)
	createNetwork.createExperiment(ds, ['gray_227_invert'], 'channel2', 'gray_invert', **tr)
	createNetwork.createExperiment(ds, ['color_227_invert'], 'channel2', 'color_invert', **tr)
	createNetwork.createExperiment(ds, ['gray_227'], 'channel2', 'gray', **tr)


def variantExperiments():
	group = "variants_2"
	
	experiment = {"combo_gbGB": (generateTag('gbGB'), dict(num_experiments=10, **default)),
				  "combo_gG": (generateTag('gG'), dict(num_experiments=10, **default))}

	for name, (t, tr) in experiment.items():
		createNetwork.createExperiment(ds, t, group, name, **tr)

def channelExperiments():
	group = "variants"
	#tags = ["gray_227","gray_227_invert", "binary_227", "binary_227_invert", "color_227", "color_227_invert"]
	#tags = {"combo_gG": [TAGS['g'], TAGS['G']], "combo_bB": [TAGS['b'], TAGS['B']], "combo_cC": [TAGS['c'], TAGS['C']],
	#		"combo_gb": [TAGS['g'], TAGS['b']], "combo_GB": [TAGS['G'], TAGS['B']], "combo_gbGB": [TAGS['g'], TAGS['b'], TAGS['B'], TAGS['G']],
	#		"combo_cg": [TAGS['c'], TAGS['g']], "combo_cb": [TAGS['c'], TAGS['b']], "combo_cgb": [TAGS['c'], TAGS['g'], TAGS['b']],
	#		"combo_CG": [TAGS['C'], TAGS['G']], "combo_CB": [TAGS['C'], TAGS['B']], "combo_CGB": [TAGS['C'], TAGS['G'], TAGS['B']],
	#		"combo_cgbCGB": [TAGS['c'], TAGS['g'], TAGS['b'], TAGS['C'], TAGS['G'], TAGS['B']]}

	experiments = {"combo_cgbcgb" : (generateTag('cgbcgb'), dict(num_experiments=10, **default))}



	for name, (t, tr) in experiments.items():
		createNetwork.createExperiment(ds, t, group, name, **tr)


	#transforms = [("mean_shifted",dict(shift="mean", scale=(1.0/255)))]

	#transforms = [("mean_shifted",dict(shift="mean", scale=(1.0/255))), ("zero_centered", dict(shift=127, scale=(1.0/255))), ("scaled", dict(scale=(1.0/255)))]


if __name__ == "__main__":
	#print COMBO(ds, 227, multiple=True)
	#sizeExperiments()
	#size2Experiments()
	#paddingExperiments()
	#padding2Experiments()
	padding4Experiments()
	#finetuneExperiments()
	#finetune2Experiments()
	#finetune3Experiments()
	#finetune4Experiments()
	#dsurfExperiments()
	#dsurf2Experiments()
	#bnExperiments()
	#bn2Experiments()
	#depthExperiments()
	#widthExperiments()
	#width3Experiments()
	#augmentationExperiments()
	#augmentation2Experiments()
	#augmentation3Experiments()
	#channelExperiments()
	#channel2Experiments()
	#variantExperiments()
	#hsvExperiments()
	#equiArchExperiments()
