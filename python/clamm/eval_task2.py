
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

def argmax(l):
	ele = max(l)
	return l.index(ele)

def load_predictions(f):
	d = dict()
	for line in open(f, 'r').readlines():
		tokens = line.split()
		vals = map(float, tokens[1:])
		top1 = argmax(vals)
		vals[top1] = 0
		top2 = argmax(vals)
		d[tokens[0]] = (top1, top2)
	return d

def load_gt(f, class_to_id):
	d = dict()
	for line in open(f, 'r').readlines():
		tokens = line.split()
		if len(tokens) == 2:
			d[tokens[0]] = class_to_id[tokens[1]]
		else:
			d[tokens[0]] = (class_to_id[tokens[1]], class_to_id[tokens[2]])
	return d
	

class_to_id = load_class_ids(class_mapping_file)
predictions = load_predictions(predictions_file)
gt = load_gt(gt_file, class_to_id)

assert len(predictions) == len(gt)
assert set(predictions.keys()) == set(gt.keys())

single_label_points = 0
num_single_label = 0
two_label_points = 0
num_two_labels = 0
for im_name in predictions:
	predicted_classes = predictions[im_name]
	actual_classes = gt[im_name]
	if isinstance(actual_classes, tuple):
		# two labels
		num_two_labels += 1
		if ((predicted_classes[0] == actual_classes[0] and
			 predicted_classes[1] == actual_classes[1]) or
		    (predicted_classes[0] == actual_classes[1] and
			 predicted_classes[1] == actual_classes[0])):
			# both predictions match both labels, regardless of ordering
			two_label_points += 4
		elif (predicted_classes[0] == actual_classes[0] or
		      predicted_classes[0] == actual_classes[1]):
			# only the top prediction matches one of the labels
			two_label_points += 2
		elif (predicted_classes[1] == actual_classes[0] or
		      predicted_classes[1] == actual_classes[1]):
			# only the second prediction matches one of the labels
			two_label_points += 1
		else:
			two_label_points -= 2
	else:
		# single label
		num_single_label += 1
		actual_class = actual_classes
		if actual_class == predicted_classes[0] or actual_class == predicted_classes[1]:
			single_label_points += 4
		else:
			single_label_points -= 2

total_points = single_label_points + two_label_points
total_instances = num_single_label + num_two_labels
print "Average total points: %.3f" % (total_points / float(total_instances))
print "Average two-label points: %.3f" % (two_label_points / float(num_two_labels))
print "Average single-label points: %.3f" % (single_label_points / float(num_single_label))


