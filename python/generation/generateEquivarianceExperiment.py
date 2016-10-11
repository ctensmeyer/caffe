import os
import createNetwork


########################
datasets = ['rvl_cdip']
#######################
MAPPINGS = ['identity', 'linear', 'mlp']

def process_tag(T, size=227, pad=False, multiple=False, multiple2=False):
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


def get_tags(ds, size=227, pad=False, multiple=False, multiple2=False):
    #print multiple
    if ds.startswith('rvl_cdip'):
        #tags = ['g', 'b', 'G', 'B']
        tags = ['g', 'G']
        #tags = ['g']
    else:
        #tags = ['c', 'g', 'b', 'C', 'G', 'B']
        tags = ['c', 'C']
        #tags = ['c']

    return map(lambda t: process_tag(t, size, pad=pad, multiple=multiple, multiple2=multiple2), tags)

d_l_tparams = {
	'mirror': [dict(), 
			   dict(hmirror=1), 
			   dict(vmirror=1), 
			   dict(hmirror=1, vmirror=1)
			  ],
	'color': [dict(), 
			  dict(color=[-15, 1]), 
			  dict(color=[-10, 1]), 
			  dict(color=[-5, 1]), 
			  dict(color=[0, 1]), 
			  dict(color=[5, 1]), 
			  dict(color=[10, 1]), 
			  dict(color=[15, 1]), 
			 ],
	'noise': [dict(), 
			  dict(noise=[3, 4]), 
			  dict(noise=[7, 8]), 
			  dict(noise=[11, 12]), 
			  dict(noise=[15, 16]), 
			  dict(noise=[19, 20]), 
			 ],
	'perspective': [dict(), 
			  		dict(perspective=[-5.20e-05, -4.90e-05, -5.12e-05, 5.07e-05, 4.55e-05, -5.06e-05, 4.74e-05, 4.60e-05]),
			  		dict(perspective=[-6.13e-05, 6.49e-05, 6.25e-05, -6.64e-05, 6.62e-05, 6.62e-05, -6.50e-05, -6.64e-05]),
			  		dict(perspective=[-0.000140, -0.000139, 0.0001359, -0.000135, -0.000136, -0.000138, -0.000137, -0.000137]),
			  		dict(perspective=[-0.000144, -0.000147, -0.000145, 0.000147, 0.000148, -0.000143, -0.000145, -0.000149]),
			  		dict(perspective=[0.000155, -0.000151, -0.000154, -0.000153, 0.000152, 0.000151, -0.000156, 0.000152])
				   ],
	'rotation': [dict(), 
			  	 dict(rotation=[4,5,0]), 
			  	 dict(rotation=[4,5,1]), 
			  	 dict(rotation=[9,10,0]), 
			  	 dict(rotation=[9,10,1]), 
			  	 dict(rotation=[14,15,0]), 
			  	 dict(rotation=[14,15,1]), 
			  	 dict(rotation=[19,20,0]), 
			  	 dict(rotation=[19,20,1]), 
			    ],
	'shear': [dict(), 
			  dict(shear=[4,5,0,0]), 
			  dict(shear=[4,5,0,1]), 
			  dict(shear=[4,5,1,0]), 
			  dict(shear=[4,5,1,1]), 
			  dict(shear=[9,10,0,0]), 
			  dict(shear=[9,10,0,1]), 
			  dict(shear=[9,10,1,0]), 
			  dict(shear=[9,10,1,1]), 
			  dict(shear=[14,15,0,0]), 
			  dict(shear=[14,15,0,1]), 
			  dict(shear=[14,15,1,0]), 
			  dict(shear=[14,15,1,1]), 
			  dict(shear=[19,20,0,0]), 
			  dict(shear=[19,20,0,1]), 
			  dict(shear=[19,20,1,0]), 
			  dict(shear=[19,20,1,1]), 
			 ],
	'elastic': [dict(), 
			  	dict(elastic=[2,4,5]), 
			  	dict(elastic=[2,9,10]), 
			  	dict(elastic=[2.5,4,5]), 
			  	dict(elastic=[2.5,9,10]), 
			  	dict(elastic=[3,4,5]), 
			  	dict(elastic=[3,9,10]), 
			  	dict(elastic=[3.5,4,5]), 
			  	dict(elastic=[3.5,9,10]), 
			   ],
	'blur': [dict(), 
			 dict(blur=[0.4, 0.5]), 
			 dict(blur=[0.9, 1.0]), 
			 dict(blur=[1.4, 1.5]), 
			 dict(blur=[1.9, 2.0]), 
			 dict(blur=[2.4, 2.5]), 
			 dict(blur=[2.9, 3.0]), 
			],
}
crop_tparams = [dict(crop=['center']), 
             	dict(crop=['ul']), 
            	dict(crop=['ur']), 
             	dict(crop=['bl']), 
             	dict(crop=['br']),
			   ]

def equivarianceCropExperiments(ds):
	group = "equivariance"
	tags = get_tags(ds, size=256)

	for mapping in MAPPINGS:
		for loss in [1.]:
			name = "crop256_%s_%d" % (mapping, loss)
			print "createEquivarianceExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
			createNetwork.createEquivarianceExperiment(ds, tags, group, name, num_experiments=1, mapping=mapping, 
				l_tparams=crop_tparams, ce_loss_weight=loss, l2_loss_weight=50*loss)

def equivarianceExperiments(ds):
	group = "equivariance"
	tags = get_tags(ds)

	for name_base, l_tparams in d_l_tparams.iteritems():
		for mapping in MAPPINGS:
			#for loss in [1., 2.]:
			for loss in [1.]:
				name = "%s_%s_%d" % (name_base, mapping, loss)
				print "createEquivarianceExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
				createNetwork.createEquivarianceExperiment(ds, tags, group, name, num_experiments=1, mapping=mapping, 
					l_tparams=l_tparams, ce_loss_weight=loss, l2_loss_weight=50*loss)


if __name__ == "__main__":
	for ds in datasets:
		equivarianceExperiments(ds)
		equivarianceCropExperiments(ds)

