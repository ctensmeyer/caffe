
import os
import sys
import cv2
import math
import random
import argparse
import numpy as np
import scipy.ndimage
import traceback
from utils import get_transforms, apply_all_transforms


def main(in_file, transforms_file, out_dir):
	im = cv2.imread(in_file, 1)
	transforms, _ = get_transforms(transforms_file)
	ims = apply_all_transforms(im, transforms)
	idx = 0
	for im, transform in zip(ims, transforms):
		out_path = os.path.join(out_dir, str(idx) + '_' + transform.replace(' ', '_') + ".png")
		cv2.imwrite(out_path, im)
		idx += 1
		

if __name__ == "__main__":
	in_file = sys.argv[1]
	transforms_file = sys.argv[2]
	out_dir = sys.argv[3]
	main(in_file, transforms_file, out_dir)

