
import random

def make_single_param_transforms(name, param_min, param_max, num_transforms, include_whole=True, exclude_zero=True):
	transforms = list()
	if include_whole:
		transforms.append("none")

	step = (param_max - param_min) / float(num_transforms)
	cur = param_min

	for n in xrange(num_transforms):
		if exclude_zero and abs(cur) < 0.000001:
			cur += step
		transform = "%s %f" % (name, cur)
		cur += step
                transforms.append(transform)

	return transforms

def make_crop_transforms(image_size, crop_size, grid_size, include_whole=True):
	'''
	Returns a list of strings representing an evenly spaced 2D grid of 
		square crops of size $crop_size for a square image of size $image_size.
		The number of crops in each dimension is $grid_size.
	'''
	transforms = list()
	if include_whole:
		transforms.append("resize %d %d" % (crop_size, crop_size))
	d = image_size - crop_size
	for y in xrange(grid_size):
		off_y = y * d / float(grid_size - 1)
		for x in xrange(grid_size):
			off_x = x * d / (grid_size - 1)
			transform = "crop %d %d %d %d" % (off_y, off_x, crop_size, crop_size)
			transforms.append(transform)
	return transforms

def make_param_random_transforms(prefix, max_param, num_transforms, seed=54321, include_whole=True):
	transforms = make_single_param_transforms(prefix, 0, max_param, num_transforms, include_whole)
	random.seed(seed)
	for n in xrange(1 if include_whole else 0, num_transforms + (1 if include_whole else 0)):
		transforms[n] = transforms[n] + " %d" % random.randint(2, 999999999)
	return transforms

def make_gaussnoise_transforms(max_sigma, num_transforms, seed=54321, include_whole=True):
	return make_param_random_transforms("gaussnoise", max_sigma, num_transforms, seed, include_whole)

def make_color_jitter_transforms(sigma, num_transforms, seed=54321, include_whole=True):
	return make_param_random_transforms("color_jitter", sigma, num_transforms, seed, include_whole)

def make_perspective_transforms(max_sigma, num_transforms, seed=3141519, include_whole=True):
	transforms = list()
	if include_whole:
		transforms.append("none")

	random.seed(seed)
	step = max_sigma / num_transforms
	cur = step
	for n in xrange(num_transforms):
		l = list()
		for x in xrange(8):
			# divide by 2 so that 95% of values are below cur
			l.append(random.gauss(0, cur / 2))
		transform = "perspective %s" % " ".join(map(str, l))
		transforms.append(transform)
		cur += step

	return transforms

def make_elastic_deformation_transforms(sigma, max_alpha, num_transforms, include_whole=True):
	return make_param_random_transforms("elastic %f" % sigma, 0, max_alpha, num_transforms, include_whole)
	
def make_rotation_transforms(max_angle, num_transforms, include_whole=True):
	return make_single_param_transforms("rotation", -1* max_angle, max_angle, num_transforms, include_whole)

def make_blur_transforms(max_sigma, num_transforms, include_whole=True):
	return make_single_param_transforms("blur", 0, max_sigma, num_transforms, include_whole)

def make_shear_transforms(max_angle, num_transforms, include_whole=True):
	transforms = make_single_param_transforms("shear", -1* max_angle, max_angle, num_transforms, include_whole)
	for n in xrange(1 if include_whole else 0, num_transforms + (1 if include_whole else 0)):
		if n % 2 == 0:
			transforms[n] = transforms[n] + " h"
		else:
			transforms[n] = transforms[n] + " v"
	return transforms

def make_unsharp_transforms(max_sigma, num_transforms, min_amount=0.25, max_amount=2, num_amounts=7, include_whole=True):
	transforms = make_single_param_transforms("unsharp", 0, max_sigma, num_transforms, include_whole)
	
	amount_step = (max_amount - min_amount) / num_amounts
	cur_amount = min_amount

	for n in xrange(1 if include_whole else 0, num_transforms + (1 if include_whole else 0)):
		transforms[n] = transforms[n] + " %f" % cur_amount

		cur_amount += amount_step
		if cur_amount > max_amount:
			cur_amount = min_amount

	return transforms


def make_mirror_transforms(h,v, include_whole=True):
    h = h>0
    v = v>0
    
    transforms = list()
    if include_whole:
        transforms.append("none")

    if h:
        transforms.append("mirror h")

    if v:
        transforms.append("mirror v")

    if h and v:
        transforms.append("mirror hv")


    return transforms

def cross_mirror(transforms, h, v):
	new_transforms = list()
	new_transforms += transforms
	if h:
		new_transforms += [line + ";mirror h" for line in transforms]
	if v:
		new_transforms += [line + ";mirror v" for line in transforms]
	if h and v:
		new_transforms += [line + ";mirror hv" for line in transforms]

	return new_transforms

def cross_shear(transforms, max_angle, num_repeats=1):
	new_transforms = list()
	new_transforms += transforms

	for line in transforms:
		for _ in xrange(num_repeats):
			angle = random.uniform(-max_angle, max_angle)
			orien = 'h' if random.random() > 0.5 else 'v'
			new_line = "shear %.3f %s;" % (angle, orien) + line
			new_transforms.append(new_line)
	return new_transforms


def make_crop_shear_mirror_transforms(im_size, crop_size, h, v, max_angle, shear_repeats):
	transforms = make_crop_transforms(im_size, crop_size, 3, include_whole=True)
	transforms = cross_mirror(transforms, h, v)
	transforms = cross_shear(transforms, max_angle, shear_repeats)
	return transforms

def make_crop_shear_transforms(im_size, crop_size, max_angle, shear_repeats):
	transforms = make_crop_transforms(im_size, crop_size, 3, include_whole=True)
	transforms = cross_shear(transforms, max_angle, shear_repeats)
	return transforms

def make_crop_mirror_transforms(im_size, crop_size, h, v):
	transforms = make_crop_transforms(im_size, crop_size, 3, include_whole=True)
	transforms = cross_mirror(transforms, h, v)
	return transforms

def make_shear_mirror_transforms(h, v, max_angle, shear_repeats):
	transforms = make_mirror_transforms(h, v)
	transforms = cross_shear(transforms, max_angle, shear_repeats)
	return transforms

