import os
import createNetwork

DS = ["balinese1_color", "balinese2_color", "cmater_real_color", "hbr_color", 
	"hdibco", "icfhr2010", 'iupr', 'parzival', 'phidb', 'rodrigo', 'saintgall_color']

########################
ds = 'saintgall_color'
#######################

TAG_SETS = {
			"original": ['original_images']
}

def depthExperiments():
	group = "depth"
	tags = TAG_SETS["original"]


	for depth in [0, 1, 2, 3, 4, 5]:
		name = "depth_%d" % depth
		print "createBinarizeExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, {'num_experiments':2, 'depth':depth})
		createNetwork.createBinarizeExperiment(ds, tags, group, name, num_experiments=2, depth=depth, wfm_loss=True, lr=0.00001)


if __name__ == "__main__":
	depthExperiments()

