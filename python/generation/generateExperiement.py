import os
import createNetwork

DS = ["rvl_cdip", "andoc_1m", "rvl_cdip_10", "rvl_cdip_100", "andoc_1m_10", "andoc_1m_50", "imagenet"]

########################
#ds = 'rvl_cdip_100'
ds = 'andoc_1m'
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


SIZES = [32, 64, 100, 150, 227, 256, 384, 512]
MULTISCALE_SIZES = [(150, 227, 256), (227, 256, 320), (256, 320, 384), (320, 384, 512), (227, 256, 320, 384, 512)]
WIDTHS = [0.1, 0.25, 0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 2]
DEPTHS = [0, 1, 2, 3, 4, 5, 6]
default = dict(shift="mean", scale=(1.0/255))
width_default = dict(shift="mean", scale=(1.0/255), shear=10, num_experiments=2)
size_default = dict(shift="mean", scale=(1.0/255), shear=10, num_experiments=2)
pad_default = dict(shift="mean", scale=(1.0/255), shear=10, num_experiments=2)

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
                               
                               "sharp_1_5":   (COMBO(ds), dict(unsharp=1.5, **default)),
                               "sharp_3":   (COMBO(ds), dict(unsharp=3, **default)),
                               
                               "sharp_blur_1_5":   (COMBO(ds), dict(blur=1.5, unsharp=1.5, **default)),
                               "sharp_blur_3":   (COMBO(ds), dict(blur=3, unsharp=3, **default))
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
                "combined": { "crop_mirror_shear": (COMBO(ds, 256), dict(crop=227, hmirror=0.0, vmirror=0.5, shear=10, **default)),
                              "crop_mirror": (COMBO(ds, 256), dict(crop=227, hmirror=0.0, vmirror=0.5, **default)),
                              "mirror_shear": (COMBO(ds), dict(hmirror=0.0, vmirror=0.5, shear=10, **default)),
                              "crop_shear": (COMBO(ds, 256), dict(crop=227, shear=10, **default)),
                           },

                "size": { "size_%d" % size: (COMBO(ds, size), dict(**size_default)) for size in SIZES },
                "size2": { "size_320": (COMBO(ds, 320), dict(**size_default))},
                "multiscale": { "multiscale_%d_%d" % (sizes[0], sizes[-1]): (COMBO(ds, sizes[-1]), dict(pool='spp', **size_default)) for sizes in MULTISCALE_SIZES },

                "width": { "width_%d" % int(width * 100): (COMBO(ds), dict(width_mult=width, **width_default)) for width in WIDTHS },
                "width_2": { "width_conv_50" : (COMBO(ds), dict(conv_width_mult=0.50, **width_default)),
                             "width_conv_75" : (COMBO(ds), dict(conv_width_mult=0.75, **width_default)),
                             "width_fc_50" : (COMBO(ds), dict(fc_width_mult=0.50, **width_default)),
                             "width_fc_75" : (COMBO(ds), dict(fc_width_mult=0.75, **width_default)),
                           },
                "padding": { 
                             "warped_227": (COMBO(ds, 227), dict(**pad_default)),
                             "pad_227":    (COMBO(ds, 227, pad=True), dict(**pad_default)),
                             "warped_384": (COMBO(ds, 384), dict(**pad_default)),
                             "pad_384":    (COMBO(ds, 384, pad=True), dict(**pad_default)),

                             "spp_warped_227": (COMBO(ds, 227), dict(pool='spp', **pad_default)),
                             "spp_pad_227":    (COMBO(ds, 227, pad=True), dict(pool='spp', **pad_default)),
                             "spp_warped_384": (COMBO(ds, 384), dict(pool='spp', **pad_default)),
                             "spp_pad_384":    (COMBO(ds, 384, pad=True), dict(pool='spp', **pad_default)),

                             "hvp_warped_227": (COMBO(ds, 227), dict(pool='hvp', **pad_default)),
                             "hvp_pad_227":    (COMBO(ds, 227, pad=True), dict(pool='hvp', **pad_default)),
                             "hvp_warped_384": (COMBO(ds, 384), dict(pool='hvp', **pad_default)),
                             "hvp_pad_384":    (COMBO(ds, 384, pad=True), dict(pool='hvp', **pad_default)),
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
                "depth": { "depth_%d" % depth: (COMBO(ds), dict(depth=depth, width_mult=0.25, **width_default)) for depth in DEPTHS },
                }


def sizeExperiments():
    group = "size"

    experiments = EXPERIMENTS["multiscale"]

    for name, (tags, tr) in experiments.items():
        print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
        createNetwork.createExperiment(ds, tags, group, name, **tr)

def depthExperiments():
    group = "depth2"

    experiments = EXPERIMENTS["depth"]

    for name, (tags, tr) in experiments.items():
        print "createNetwork.createExperiment(%r, %r, %r, %r, %r)" % (ds, tags, group, name, tr)
        createNetwork.createExperiment(ds, tags, group, name, **tr)


def paddingExperiments():
    group = "padding"

    #experiments = EXPERIMENTS["padding"]
    #experiments = EXPERIMENTS["multiple2"]
    #experiments.update(EXPERIMENTS["multiple"])
    experiments = EXPERIMENTS["multiple"]

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

def augmentationExperiments():
    group = "augmentation"
    
    experiments = EXPERIMENTS["standard"]
    #experiments = EXPERIMENTS['baseline']
    #experiments.update(EXPERIMENTS["standard"])
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
    #        "combo_gb": [TAGS['g'], TAGS['b']], "combo_GB": [TAGS['G'], TAGS['B']], "combo_gbGB": [TAGS['g'], TAGS['b'], TAGS['B'], TAGS['G']],
    #        "combo_cg": [TAGS['c'], TAGS['g']], "combo_cb": [TAGS['c'], TAGS['b']], "combo_cgb": [TAGS['c'], TAGS['g'], TAGS['b']],
    #        "combo_CG": [TAGS['C'], TAGS['G']], "combo_CB": [TAGS['C'], TAGS['B']], "combo_CGB": [TAGS['C'], TAGS['G'], TAGS['B']],
    #        "combo_cgbCGB": [TAGS['c'], TAGS['g'], TAGS['b'], TAGS['C'], TAGS['G'], TAGS['B']]}

    experiments = {"combo_cgbcgb" : (generateTag('cgbcgb'), dict(num_experiments=10, **default))}



    for name, (t, tr) in experiments.items():
        createNetwork.createExperiment(ds, t, group, name, **tr)


    #transforms = [("mean_shifted",dict(shift="mean", scale=(1.0/255)))]

    #transforms = [("mean_shifted",dict(shift="mean", scale=(1.0/255))), ("zero_centered", dict(shift=127, scale=(1.0/255))), ("scaled", dict(scale=(1.0/255)))]


if __name__ == "__main__":
    #print COMBO(ds, 227, multiple=True)
    #sizeExperiments()
    paddingExperiments()
    #depthExperiments()
    #widthExperiments()
    #augmentationExperiments()
    #channelExperiments()
    #variantExperiments()
