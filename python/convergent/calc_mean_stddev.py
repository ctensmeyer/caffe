
import numpy as np
import sys
import os
import errno

def safe_mkdir(_dir):
	try:
		os.makedirs(os.path.join(_dir))
	except OSError as exc:
		if exc.errno != errno.EEXIST:
			raise

in_dir = sys.argv[1]
centered_dir = sys.argv[2]
normalized_dir = sys.argv[3]

safe_mkdir(centered_dir)
safe_mkdir(normalized_dir)

for fname in os.listdir(in_dir):
	if fname == "labels.txt":
		continue
	arr = np.loadtxt(os.path.join(in_dir, fname))
	col_means = np.mean(arr, axis=0)
	col_std = np.std(arr, axis=0)

	arr = arr - col_means[np.newaxis, :]
	np.savetxt(os.path.join(centered_dir, fname), arr, "%7.4f")

	arr = arr / col_std[np.newaxis, :]
	np.savetxt(os.path.join(normalized_dir, fname), arr, "%7.4f")
	del arr

