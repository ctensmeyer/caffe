import os
import createNetwork


DATASET_TAGS={"andoc_1m": ["binary_227", "color_150", "color_256", "color_384", "color_64",
                           "binary_227_invert", "color_227", "color_256_padded", "color_384_padded",
                           "gray_227", "color_100", "color_227_invert", "color_32", "color_512", "gray_227_invert"],

              "rvl_cdip":  ["binary_227", "gray_150", "gray_256", "gray_384", "gray_64",
                           "binary_227_invert", "gray_227", "gray_256_padded", "gray_384_padded",
                           "gray_100", "gray_227_invert", "gray_32", "gray_512"]
            }



def TAGS(T, size=227):
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

    return tag


def generateTag(T, size=227):
    return map(lambda t: TAGS(t, size), T)

#TAGS = { "b": "binary_227", "g": "gray_227", "B": "binary_227_invert", "G": "gray_227_invert", "c": "color_227", "C": "color_227_invert"}


DS = ["rvl_cdip", "andoc_1m"]

########################
ds = DS[0]
#######################

def COMBO(ds=ds, size = 227):
    if ds == DS[0]:
        #tags = ['g', 'b', 'G', 'B']
        tags = ['g', 'G']
    else:
        #tags = ['c', 'g', 'b', 'C', 'G', 'B']
        tags = ['c', 'C']


    return map(lambda t: TAGS(t, size), tags)


#COMBO = [TAGS['g'], TAGS['b'], TAGS['G'], TAGS['B']]


default = dict(shift="mean", scale=(1.0/255))

EXPERIMENTS = {"standard": {"h_mirror" : (COMBO(ds), dict(hmirror=0.5, **default)),
                            "v_mirror" : (COMBO(ds), dict(vmirror=0.5, **default)),
                            "hv_mirror" : (COMBO(ds), dict(hmirror=0.5, vmirror=0.5, **default)),
                   
                            "gauss_noise_5" : (COMBO(ds), dict(noise_std=[0,5], **default)),
                            "gauss_noise_10" : (COMBO(ds), dict(noise_std=[0,10], **default)),
                            "gauss_noise_15" : (COMBO(ds), dict(noise_std=[0,15], **default)),
                            "gauss_noise_20" : (COMBO(ds), dict(noise_std=[0,20], **default)),
                   
                            "crop_240" : (COMBO(ds,240), dict(crop=True, **default)),
                            "crop_256" : (COMBO(ds,256), dict(crop=True, **default)),
                            "crop_288" : (COMBO(ds,288), dict(crop=True, **default)),
                            "crop_320" : (COMBO(ds,320), dict(crop=True, **default))
                            },
                
                "rotate_shear":  {"rotation_5": (COMBO(ds), dict(rotation=5, **default)),
                                  "rotation_10": (COMBO(ds), dict(rotation=10, **default)),
                                  "rotation_15": (COMBO(ds), dict(rotation=15, **default)),
                                  "rotation_20": (COMBO(ds), dict(rotation=20, **default)),
                   
                                  "shear_5": (COMBO(ds), dict(shear=5, **default)),
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
								}

                }

def augmentationExperiments():
    group = "augmentation_2"
    ds = DS[0]
    
    #experiments = EXPERIMENTS['standard']
    experiments = EXPERIMENTS['standard'].copy()
    #experiments.update(EXPERIMENTS["rotate_shear"])
    #experiments.update(EXPERIMENTS["blur_sharp"])
    #experiments.update(EXPERIMENTS["perspective"])
    experiments.update(EXPERIMENTS["color_jitter"])

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
    augmentationExperiments()
    #channelExperiments()
    #variantExperiments()
