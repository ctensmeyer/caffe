import os
import createNetwork


########################
#datasets = ['rvl_cdip', 'imagenet']
datasets = ['imagenet']
#######################
MAPPINGS = ['identity', 'linear']

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
        #tags = ['g', 'G']
        tags = ['g']
    else:
        #tags = ['c', 'g', 'b', 'C', 'G', 'B']
        #tags = ['c', 'C']
        tags = ['c']

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
			  dict(color=[5, 1]), 
			  dict(color=[10, 1]), 
			  dict(color=[15, 1]), 
			 ],
	'noise': [dict(), 
			  dict(noise=[3.9, 4]), 
			  dict(noise=[7.9, 8]), 
			  dict(noise=[11.9, 12]), 
			  dict(noise=[15.9, 16]), 
			  dict(noise=[19.9, 20]), 
			 ],
	'perspective': [dict(), 
			  		dict(perspective=[-5.20e-05, -4.90e-05, -5.12e-05, 5.07e-05, 4.55e-05, -5.06e-05, 4.74e-05, 4.60e-05]),
			  		dict(perspective=[-6.13e-05, 6.49e-05, 6.25e-05, -6.64e-05, 6.62e-05, 6.62e-05, -6.50e-05, -6.64e-05]),
			  		dict(perspective=[-0.000140, -0.000139, 0.0001359, -0.000135, -0.000136, -0.000138, -0.000137, -0.000137]),
			  		dict(perspective=[-0.000144, -0.000147, -0.000145, 0.000147, 0.000148, -0.000143, -0.000145, -0.000149]),
			  		dict(perspective=[0.000155, -0.000151, -0.000154, -0.000153, 0.000152, 0.000151, -0.000156, 0.000152])
				   ],
	'rotation': [dict(), 
			  	 dict(rotation=[4,6,0]), 
			  	 dict(rotation=[4,6,1]), 
			  	 dict(rotation=[9,11,0]), 
			  	 dict(rotation=[9,11,1]), 
			  	 dict(rotation=[14,16,0]), 
			  	 dict(rotation=[14,16,1]), 
			    ],
	'shear': [dict(), 
			  dict(shear=[4,6,0,0]), 
			  dict(shear=[4,6,1,0]), 
			  dict(shear=[9,11,0,1]), 
			  dict(shear=[9,11,1,1]), 
			  dict(shear=[14,16,1,0]), 
			  dict(shear=[19,21,1,1]), 
			 ],
	'elastic': [dict(), 
			  	dict(elastic=[2.5,4.9,5]), 
			  	dict(elastic=[2.5,9.9,10]), 
			  	dict(elastic=[3,4.9,5]), 
			  	dict(elastic=[3,9.9,10]), 
			  	dict(elastic=[3.5,4.9,5]), 
			  	dict(elastic=[3.5,9.9,10]), 
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

def equivarianceTestExperiments(ds):
	group = "equivariance_test2"
	tags = get_tags(ds)

	rotate_params = d_l_tparams['rotation']
	for mapping in ['linear']:
		for loss in [0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.]:
			name = "loss_%s_%.2f" % (mapping, loss)
			print "createEquivarianceExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
			createNetwork.createEquivarianceExperiment(ds, tags, group, name, num_experiments=1, mapping=mapping, 
				l_tparams=rotate_params, ce_loss_weight=loss, l2_loss_weight=50*loss, batch_size=10)


	#for mapping in ['linear']:
	#	for batch in [8, 16, 32]:
	#		name = "rotate_batch_%s_%d" % (mapping, batch)
	#		print "createEquivarianceExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	#		createNetwork.createEquivarianceExperiment(ds, tags, group, name, num_experiments=1, mapping=mapping, 
	#			l_tparams=[dict(rotation=[0, 15, 0.5])], batch_size=batch, ce_loss_weight=loss, l2_loss_weight=50*loss)

	#for mapping in ['linear']:
	#	for batch in [8, 16, 32]:
	#		name = "base_batch_%s_%d" % (mapping, batch)
	#		print "createEquivarianceExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
	#		createNetwork.createEquivarianceExperiment(ds, tags, group, name, num_experiments=1, mapping=mapping, 
	#			l_tparams=[{}], batch_size=batch, ce_loss_weight=loss, l2_loss_weight=50*loss)

	for mapping in ['linear']:
		for loss in [0.3]:
			for max_rotation in [2.5, 5, 10, 15, 20]:
				name = "variable_%s_%d" % (mapping, max_rotation)
				print "createEquivarianceExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
				params = list(rotate_params)
				params[0] = dict(rotation=[0, max_rotation, 0.5])
				createNetwork.createEquivarianceExperiment(ds, tags, group, name, num_experiments=1, mapping=mapping, 
					l_tparams=params, ce_loss_weight=loss, l2_loss_weight=50*loss, batch_size=10)

	rotate_params = d_l_tparams['rotation']
	loss = 0
	for mapping in ['identity']:
		for degree in [5, 10, 15, 20]:
			name = "measure_%d" % degree
			print "createEquivarianceExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
			params = list(rotate_params)
			params[0] = dict(rotation=[0, degree, 0.5])
			createNetwork.createEquivarianceExperiment(ds, tags, group, name, num_experiments=1, mapping=mapping, 
				l_tparams=params, ce_loss_weight=loss, l2_loss_weight=50*loss, batch_size=10)

def equivarianceCropExperiments(ds):
	group = "equivariance2"
	tags = get_tags(ds, 256)

	#for mapping in MAPPINGS:
	for mapping in ['linear']:
	#for mapping in ['identity']:
		for loss in [.3]:
			name = "crop256_%s_%.1f" % (mapping, loss)
			print "createEquivarianceExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
			createNetwork.createEquivarianceExperiment(ds, tags, group, name, num_experiments=1, mapping=mapping, 
				l_tparams=crop_tparams, ce_loss_weight=loss, l2_loss_weight=50*loss, batch_size=10)

def equivarianceExperiments(ds):
	group = "equivariance2"
	tags = get_tags(ds)

	for name_base, l_tparams in d_l_tparams.iteritems():
		#for mapping in MAPPINGS:
		for mapping in ['linear']:
		#for mapping in ['identity']:
			for loss in [0.3]:
				name = "%s_%s_%.1f" % (name_base, mapping, loss)
				print "createEquivarianceExperiment(%r, %r, %r, %r)" % (ds, tags, group, name)
				createNetwork.createEquivarianceExperiment(ds, tags, group, name, num_experiments=1, mapping=mapping, 
					l_tparams=l_tparams[:6], ce_loss_weight=loss, l2_loss_weight=50*loss, batch_size=10)


if __name__ == "__main__":
	for ds in datasets:
		equivarianceExperiments(ds)
		#equivarianceCropExperiments(ds)
		#equivarianceTestExperiments(ds)

