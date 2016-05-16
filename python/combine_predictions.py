
import sys
import pickle
import numpy as np


def same_len(ll):
	lens = map(len, ll)
	return all(map(lambda _len: _len == lens[0], lens))

def same_val(ll, idx):
	vals = map(lambda x: x[idx], ll)
	return all(map(lambda val: val == vals[0], vals))

all_labels = list()
all_mean_outputs = list()

for fn in sys.argv[1:]:
	fd = open(fn, 'rb')
	mean_outputs, labels = pickle.load(fd)
	all_labels.append(labels)
	all_mean_outputs.append(mean_outputs)

if not same_len(all_labels + all_mean_outputs):
	print "error, some things are not the same size\nLens:"
	print map(len, all_labels + all_mean_outputs)
	exit(1)

num_correct = 0
num_total = len(all_labels[0])

for idx in xrange(num_total):
	if not same_val(all_labels, idx):
		print "At index %d, the labels differ" % idx
		exit(1)
	label = all_labels[0][idx]
	instance_outputs = map(lambda l: l[idx], all_mean_outputs)
	combined = np.asarray(instance_outputs)
	average = np.mean(combined, axis=0)
	predicted_label = np.argmax(average) 
	if label == predicted_label:
		num_correct += 1
	
print "Accuracy: ", num_correct / float(num_total)


