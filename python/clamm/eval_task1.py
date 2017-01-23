
import os
import sys
import numpy as np

predictions_file = sys.argv[1]
gt_file = sys.argv[2]
class_mapping_file = sys.argv[3]

def load_class_ids(f):
	d = dict()
	for line in open(f, 'r').readlines():
		tokens = line.split()
		d[tokens[0]] = int(tokens[1])
	return d


def load_predictions(f):
	d = dict()
	for line in open(f, 'r').readlines():
		tokens = line.split()
		d[tokens[0]] = int(tokens[1])
	return d

def load_gt(f, class_to_id):
	d = dict()
	for line in open(f, 'r').readlines():
		tokens = line.split()
		d[tokens[0]] = class_to_id[tokens[1]]
	return d
	

class_to_id = load_class_ids(class_mapping_file)
predictions = load_predictions(predictions_file)
gt = load_gt(gt_file, class_to_id)

assert len(predictions) == len(gt)
assert set(predictions.keys()) == set(gt.keys())

correct = 0
conf_mat = np.zeros(shape=(13,13))
for im_name in predictions:
	predicted_class = predictions[im_name]
	actual_class = gt[im_name]
	if predicted_class == actual_class:
		correct += 1
	conf_mat[actual_class,predicted_class] += 1

print "Accuracy: %2.4f%%  (%d / %d)" % (100. * correct / len(gt), correct, len(gt))
print conf_mat

