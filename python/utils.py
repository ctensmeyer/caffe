
import numpy as np
import cv2
import math
import os
import errno
import scipy.ndimage


def safe_mkdir(_dir):
	try:
		os.makedirs(os.path.join(_dir))
	except OSError as exc:
		if exc.errno != errno.EEXIST:
			raise


def apply_salt(im, tokens):
	perc_pixels, perc_salt, seed = float(tokens[1]), float(tokens[2]), float(tokens[3])
	np.random.seed(seed)

	flip_map = np.random.uniform(0, 1, im.shape)
	salt_map = np.random.uniform(0, 1, im.shape)

	out = np.copy(im)
	out[np.logical_and(flip_map < perc_pixels, salt_map <= salt_perc)] = 255  # salt
	out[np.logical_and(flip_map < perc_pixels and salt_map > salt_perc)] = 0     # pepper

	return out


def apply_elastic_deformation(im, tokens):
	sigma, alpha, seed = float(tokens[1]), float(tokens[2]), int(tokens[3])
	np.random.seed(seed)

	displacement_x = np.random.uniform(-1 * alpha, alpha, im.shape[:2])
	displacement_y = np.random.uniform(-1 * alpha, alpha, im.shape[:2])

	displacement_x = scipy.ndimage.gaussian_filter(displacement_x, sigma, truncate=2)
	displacement_y = scipy.ndimage.gaussian_filter(displacement_y, sigma, truncate=2)

	coords_y = np.asarray( [ [y] * im.shape[1] for y in xrange(im.shape[0]) ])
	coords_y = np.clip(coords_y + displacement_y, 0, im.shape[0])

	coords_x = np.transpose(np.asarray( [ [x] * im.shape[0] for x in xrange(im.shape[1]) ] ))
	coords_x = np.clip(coords_x + displacement_x, 0, im.shape[1])

	# the backwards mapping function, which assures that all coords are in
	# the range of the input
	if im.ndim == 3:
		coords_y = coords_y[:,:,np.newaxis]
		coords_y = np.concatenate(im.shape[2] * [coords_y], axis=2)[np.newaxis,:,:,:]

		coords_x = coords_x[:,:,np.newaxis]
		coords_x = np.concatenate(im.shape[2] * [coords_x], axis=2)[np.newaxis,:,:,:]

		coords_d = np.zeros_like(coords_x)
		for x in xrange(im.shape[2]):
			coords_d[:,:,:,x] = x
		coords = np.concatenate( (coords_y, coords_x, coords_d), axis=0)
	else:
		coords = np.concatenate( (coords_y[np.newaxis,:,:], coords_x[np.newaxis,:,:]), axis=0)

	## first order spline interpoloation (bilinear?) using the backwards mapping
	output = scipy.ndimage.map_coordinates(im, coords, order=1, mode='reflect')

	return output


# "crop y x height width"
def apply_crop(im, tokens):
	y, x = int(tokens[1]), int(tokens[2])
	height, width = int(tokens[3]), int(tokens[4])
	if y >= im.shape[0] or x >= im.shape[1]:
		raise Exception("Invalid crop: (y,x) outside image bounds (%r with %r)" % (im.shape, tokens))
	if (y < 0 and y + height >= 0) or (x < 0 and x + width >= 0):
		raise Exception("Invalid crop: negative indexing has wrap around (%r with %r)" % (im.shape, tokens))
	if (height > im.shape[0]) or (width > im.shape[1]):
		raise Exception("Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, tokens))
	if (y + height > im.shape[0]) or (x + width > im.shape[1]):
		raise Exception("Invalid crop: crop goes off edge of image (%r with %r)" % (im.shape, tokens))
	return im[y:y+height,x:x+width]


def apply_dense_crop(im, tokens):
	height, width, = int(tokens[1]), int(tokens[2])
	y_stride, x_stride, = int(tokens[3]), int(tokens[4])
	if (height > im.shape[0]) or (width > im.shape[1]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, tokens)
		exit(1)
	ims = list()
	y = 0
	while (y + height) <= im.shape[0]:
		x = 0
		while (x + width) <= im.shape[1]:
			ims.append(im[y:y+height,x:x+width])
			x += x_stride
		y += y_stride
	
	return ims

def apply_rand_crop(im, tokens):
	height, width = int(tokens[1]), int(tokens[2])
	if (height > im.shape[0]) or (width > im.shape[1]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, tokens)
		exit(1)

	y = random.randint(0, im.shape[0] - height)
	x = random.randint(0, im.shape[1] - width)
	return im[y:y+height,x:x+width]

# "resize height width"
def apply_resize(im, tokens):
	size = int(tokens[2]), int(tokens[1])
	return cv2.resize(im, size)

# "mirror {h,v,hv}"
def apply_mirror(im, tokens):
	if tokens[1] == 'h':
		return cv2.flip(im, 0)
	elif tokens[1] == 'v':
		return cv2.flip(im, 1)
	elif tokens[1] == 'hv':
		return cv2.flip(im, -1)
	else:
		print "Unrecongized mirror operation %r" % tokens
		exit(1)


def apply_shift(im, tokens):
	im = im.astype(int)  # protect against over-flow wrapping
	if im.shape == 2:
		im = im + int(tokens[1])
	else:
		for c in xrange(im.shape[2]):
			im[:,:,c] = im[:,:,c] + int(tokens[min(c+1, len(tokens) - 1)])
	im = np.clip(im, 0, 255)
	im = im.astype(np.uint8) 
	return im


def apply_color_jitter(im, tokens):
	sigma, seed = float(tokens[1]), int(tokens[2])
	np.random.seed(seed)
	im = im.astype(int)  # protect against over-flow wrapping
	if im.shape == 2:
		im = im + int(np.random.normal(0, sigma))
	else:
		for c in xrange(im.shape[2]):
			im[:,:,c] = im[:,:,c] + int(np.random.normal(0, sigma))
	
	# truncate back to image range
	im = np.clip(im, 0, 255)
	im = im.astype(np.uint8) 
	return im

# "guassnoise sigma seed"
def apply_gaussnoise(im, tokens):
	sigma, seed = float(tokens[1]), int(tokens[2])
	np.random.seed(seed)
	noise = np.random.normal(0, sigma, im.shape[:2])
	if len(im.shape) == 2:
		im = (im + noise)
	else:
		im = im + noise[:,:,np.newaxis]
	im = np.clip(im, 0, 255)
	im = im.astype(np.uint8)
	return im

# "rotate degree"
def apply_rotation(im, tokens):
	degree = float(tokens[1])
	center = (im.shape[0] / 2, im.shape[1] / 2)
	rot_mat = cv2.getRotationMatrix2D(center, degree, 1.0)
	return cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR)

# "blur sigma"
def apply_blur(im, tokens):
	sigma = float(tokens[1])
	size = int(sigma * 4 + .999)
	if size % 2 == 0:
		size += 1
	return cv2.GaussianBlur(im, (size, size), sigma)
	
# "unsharpmask sigma amount"
def apply_unsharpmask(im, tokens):
	blurred = np.atleast_3d(apply_blur(im, tokens))
	amount = float(tokens[2])
	sharp = (1 + amount) * im + (-amount * blurred)
	sharp = np.clip(sharp, 0, 255)
	return sharp

# "shear degree {h,v}"
def apply_shear(im, tokens):
	degree = float(tokens[1])
	radians = math.tan(degree * math.pi / 180)
	shear_mat = np.array([ [1, 0, 0], [0, 1, 0] ], dtype=np.float)
	if tokens[2] == 'h':
		shear_mat[0,1] = radians
	elif tokens[2] == 'v':
		shear_mat[1,0] = radians
	else:
		print "Invalid shear type: %r" % tokens
	return cv2.warpAffine(im, shear_mat, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR)

# "perspective dy1 dx1 dy2 dx2 dy3 dx3 dy4 dx4"
def apply_perspective(im, tokens):
	pts1 = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
	pts2 = np.array([[0 + float(tokens[1]) ,0 + float(tokens[2])],
					   [1 + float(tokens[3]) ,0 + float(tokens[4])],
					   [1 + float(tokens[5]) ,1 + float(tokens[6])],
					   [0 + float(tokens[7]) ,1 + float(tokens[8])]
					   ], dtype=np.float32)
	M = cv2.getPerspectiveTransform(pts1,pts2)
	return cv2.warpPerspective(im, M, (im.shape[1], im.shape[0]))


def apply_transform(im, transform_str):
	tokens = transform_str.split()
	if tokens[0] == 'crop':
		return apply_crop(im, tokens)
	if tokens[0] == 'densecrop':
		return apply_dense_crop(im, tokens)
	if tokens[0] == 'randcrop':
		return apply_rand_crop(im, tokens)
	elif tokens[0] == 'resize':
		return apply_resize(im, tokens)
	elif tokens[0] == 'mirror':
		return apply_mirror(im, tokens)
	elif tokens[0] == 'gaussnoise':
		return apply_gaussnoise(im, tokens)
	elif tokens[0] == 'rotation':
		return apply_rotation(im, tokens)
	elif tokens[0] == 'blur':
		return apply_blur(im, tokens)
	elif tokens[0] == 'unsharpmask' or tokens[0] == 'unsharp':
		return apply_unsharpmask(im, tokens)
	elif tokens[0] == 'shear':
		return apply_shear(im, tokens)
	elif tokens[0] == 'perspective':
		return apply_perspective(im, tokens)
	elif tokens[0] == 'color_jitter':
		return apply_color_jitter(im, tokens)
	elif tokens[0] == 'salt':
		return apply_salt(im, tokens)
	elif tokens[0] == 'shift':
		return apply_shift(im, tokens)
	elif tokens[0] == 'elastic':
		return apply_elastic_deformation(im, tokens)
	elif tokens[0] == 'none':
		return im
	else:
		raise Exception("Unknown transform: %r" % transform_str)


# all transforms must yield images of the same dimensions
def apply_transforms(im, multi_transform_str):
	transform_strs = multi_transform_str.split(';')
	for ts in transform_strs:
		im = apply_transform(im, ts)
	return im


def apply_all_transforms(im, transform_strs):
	ims = list()
	for ts in transform_strs:
		im_out = apply_transforms(im, ts)
		if type(im_out) is list:
			ims.extend(im_out)
		else:
			ims.append(im_out)
	return ims


def get_transforms(transform_file):
	transforms = list()
	if transform_file:
		transforms = map(lambda s: s.rstrip().lower(), open(transform_file, 'r').readlines())
	if not transforms:
		transforms.append("none")
	transforms = filter(lambda s: not s.startswith("#"), transforms)
	fixed_transforms = not any(map(lambda s: s.startswith("densecrop"), transforms))
	return transforms, fixed_transforms


