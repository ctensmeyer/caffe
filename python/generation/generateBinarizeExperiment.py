import os
import createNetwork

DS = ["balinese1_color", "balinese2_color", "cmater_real_color", "hbr_color", 
	"hdibco", "icfhr2010", 'iupr', 'parzival', 'phidb', 'rodrigo', 'saintgall_color']

########################
datasets = ['saintgall_color', 'parzival_color']
#datasets = ['saintgall_gray', 'parzival_gray']
#datasets = ['parzival_gray']
#datasets += ['combined_parzival_saintgall_color']
#datasets += ['balinese1_color', 'balinese2_color', 'hdibco_color', 'hdibco_print_color', 'hdibco_hand_color']
#datasets += ['hbr_color', 'rodrigo_color', 'iupr_color']
#datasets += ['hdibco_howe_color']
#datasets = ['hbr_color', 'rodrigo_color', 'saintgall_color', 'parzival_color']
#datasets = ['hdibco_color']
#datasets = ['rodrigo_color']
#ds = 'saintgall_color'
#ds = 'parzival_color'
#######################

TAG_SETS = {
			"original": ['original_images'],
			"singles": ['bilateral', 'canny', 'percentile', 'otsu', 'wolf'],
			"wide_window": [('mean', 'mean_9'), ('mean', 'mean_19'), ('mean', 'mean_39'), ('mean', 'mean_79'),
			               ('median', 'median_9'), ('median', 'median_19'), ('median', 'median_39'), ('median', 'median_79') ],
			"narrow_window": [('min', 'min_3'), ('min', 'min_5'), ('min', 'min_7'), ('min', 'min_9'),
							('max', 'max_3'), ('max', 'max_5'), ('max', 'max_7'), ('max', 'max_9'),
							('percentile_10', 'percentile_10_3'), ('percentile_10', 'percentile_10_5'), ('percentile_10', 'percentile_10_7'), ('percentile_10', 'percentile_10_9'),
							('percentile_25', 'percentile_25_3'), ('percentile_25', 'percentile_25_5'), ('percentile_25', 'percentile_25_7'), ('percentile_25', 'percentile_25_9'),
							('std_dev', 'std_dev_3'), ('std_dev', 'std_dev_5'), ('std_dev', 'std_dev_7'), ('std_dev', 'std_dev_9')],
			"relative_darkness": [],
}
for x in [3, 5, 7, 9]:
	for y in [5, 10, 20, 40]:
		TAG_SETS["relative_darkness"].append( ('relative_darkness/%d' % x, 'relative_darkness_%d_%d' % (x, y)) )

TAG_SETS["all"] = sum(TAG_SETS.values(), [])

def	howeFeaturesExperiments(ds):
	group = "howe"
	tags = TAG_SETS["original"] + ["howe"]

	name = "howe"
	print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':4})
	createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=4, depth=4, kernel_size=7, wfm_loss=True)

def oneConvExperiments(ds):
	group = "conv1s"
	tags = TAG_SETS["original"]

	for one_convs in [0, 1, 2]:
		name = "conv1_%d" % one_convs
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, one_convs=one_convs)


def lrnExperiments(ds):
	group = "lrn"
	tags = TAG_SETS["original"]

	for do_lrn in [0, 1, 2, 3, 4]:
		name = "lrn_%s" % do_lrn
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, lrn=do_lrn)


def zeroExperiments(ds):
	group = "zero"
	tags = TAG_SETS["original"]

	for zero in [0, 3, 6, 9, 12, 15]:
		name = "zero_%s" % zero
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, zero_border=zero)


def poolExperiments(ds):
	group = "pool"
	tags = TAG_SETS["original"]

	for pool in [0, 1, 2, 3, 4]:
		name = "pool_%s" % pool
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, pool=pool)
		

def batchSizeExperiments(ds):
	group = "batch"
	tags = TAG_SETS["original"]

	for batch_size in [1, 2, 3, 4, 6, 8, 10, 16, 20, 24]:
		name = "batch_%s" % batch_size
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':5})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=5, depth=5, kernel_size=7, wfm_loss=True, train_batch_size=batch_size)
	

def augmentExperiments(ds):
	group = "augment"
	tags = TAG_SETS["original"]

	for augmentation in ['rotate', 'shear', 'perspective', 'color_jitter', 'elastic', 'blur', 'noise']:
		name = "augment_%s" % augmentation
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, augmentation: True})
		kwargs = {augmentation: True}
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, **kwargs)


def loss2Experiments(ds):
	group = "loss2"
	tags = TAG_SETS["original"]

	# Pseudo F-measure
	name = "loss_weighted_fmeasure"
	print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'wfm_loss': True})
	createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True)

	# Uniform F-measure
	for loss_weight in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1, 1.5, 2]:
		name = "loss_uniform_fmeasure_%s" % loss_weight
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, "wfm_loss": True, 'uniform_weights': True})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, uniform_weights=True, pfm_loss_weight=loss_weight)

	# Sigmoid Cross Entropy
	for loss_weight in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]:
		name = "loss_cross_entropy_%s" % loss_weight
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, "wfm_loss": False})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=False, pfm_loss_weight=loss_weight)


def lossExperiments(ds):
	group = "loss"
	tags = TAG_SETS["original"]

	# Pseudo F-measure
	name = "loss_weighted_fmeasure"
	print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'wfm_loss': True})
	createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True)

	# Uniform F-measure
	name = "loss_uniform_fmeasure"
	print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, "wfm_loss": True, 'uniform_weights': True})
	createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, uniform_weights=True)

	# Sigmoid Cross Entropy
	name = "loss_cross_entropy"
	print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, "wfm_loss": False})
	createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=False)


def depthExperiments(ds):
	group = "depth"
	tags = TAG_SETS["original"]

	for depth in [0, 1, 2, 3, 4, 5, 6, 7]:
		name = "depth_%d" % depth
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'depth':depth})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, kernel_size=7, depth=depth, wfm_loss=True)


def denseExperiments(ds):
	group = "dense"
	tags = TAG_SETS["original"]

	for dense in [4, 8, 12, 16, 20, 24, 28, 32]:
		name = "dense_%d" % dense
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'densenet':dense})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, kernel_size=7, depth=5, num_filters=dense, densenet=True, wfm_loss=True)


def widthExperiments(ds):
	group = "width"
	tags = TAG_SETS["original"]

	for width in [6, 12, 24, 36, 48, 64, 96, 128]:
		name = "width_%d" % width
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'width':width})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True, num_filters=width)


def kernelSizeExperiments(ds):
	group = "kernel_size"
	tags = TAG_SETS["original"]


	for size in [3, 5, 7, 9, 11]:
		name = "kernel_size_%d" % size
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3, 'kernel_size':size})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, kernel_size=size, depth=5, wfm_loss=True)


def scaleExperiments(ds):
	group = "scale"
	tags = TAG_SETS["original"]

	for scale in [2, 3, 4]:
		for _global in [0, 1, 2]:
			name = "scale_%d_global_%d" % (scale, _global)
			print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':2, 'scale':scale, 'global_features':_global})
			createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=2, depth=4, wfm_loss=True, num_scales=scale, global_features=_global)


def channelExperiments(ds):
	group = "channel"
	base_tags = TAG_SETS["original"]


	for additional_tag in TAG_SETS['all']:
		if isinstance(additional_tag, basestring):
			name = "channel_%s" % additional_tag
		else:
			name = "channel_%s" % additional_tag[1]
		tags = base_tags + [additional_tag]
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True)


def channel2Experiments(ds):
	group = "channel2"
	base_tags = TAG_SETS["original"]

	# combined narrow windows
	for size in [3, 5, 7, 9]:
		tags = [(feature, '%s_%d' % (feature, size)) for feature in ['min', 'max', 'percentile_10', 'percentile_25', 'std_dev']]
		tags += [ ('relative_darkness/%d' % size,  'relative_darkness_%d_%d' % (size, thresh)) for thresh in [10, 20]]
		tags += base_tags
		name = "channel_narrow_%d" % size
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True)

	# combined narrow windows with wide_windows
	for narrow_size, wide_size  in zip([3, 5, 7, 9], [9, 19, 39, 79]):
		tags = [(feature, '%s_%d' % (feature, narrow_size)) for feature in ['min', 'max', 'percentile_10', 'percentile_25', 'std_dev']]
		tags += [ ('relative_darkness/%d' % narrow_size,  'relative_darkness_%d_%d' % (narrow_size, thresh)) for thresh in [10, 20]]
		tags += [(feature, '%s_%d' % (feature, wide_size)) for feature in ['mean', 'median']]
		tags += base_tags
		name = "channel_narrow_wide_%d" % narrow_size
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True)

	for name, tags in TAG_SETS.items():
		name = "channel_%s_all" % name
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True)


def channel3Experiments(ds):
	group = "channel3"
	base_tags = TAG_SETS["original"]

	tags = [base_tags[0],
		    'percentile', 'bilateral', 'canny',
			('relative_darkness/9', 'relative_darkness_9_10'), 
			('median', 'median_19'), 
			('median', 'median_9'),  
			('mean', 'mean_19'), 
			('min', 'min_9')
			]

	# combined narrow windows
	name = "channel_custom"
	print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':3})
	createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=3, depth=5, kernel_size=7, wfm_loss=True)


if __name__ == "__main__":
	for ds in datasets:
		#howeFeaturesExperiments(ds)
		#oneConvExperiments(ds)
		#lrnExperiments(ds)
		#augmentExperiments(ds)
		#depthExperiments(ds)
		#scaleExperiments(ds)
		#widthExperiments(ds)
		#kernelSizeExperiments(ds)
		#channelExperiments(ds)
		#channel2Experiments(ds)
		#channel3Experiments(ds)
		#lossExperiments(ds)
		#loss2Experiments(ds)
		#zeroExperiments(ds)
		denseExperiments(ds)
		#poolExperiments(ds)
		#batchSizeExperiments(ds)
