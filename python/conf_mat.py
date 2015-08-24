
import os
import traceback
import argparse
import matplotlib
matplotlib.use("cairo")
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import collections
import sklearn.metrics
import json

def parse_log_file(args):
	sequences = collections.defaultdict(list)
	for ln,line in enumerate(open(args.log_file, 'r').readlines(), start=1):
		line = line.rstrip()
		tokens = line.split()
		try:
			if len(tokens) < 9:
				continue
			if tokens[4].startswith('[') and tokens[4].endswith(']'):
				output_name = tokens[4][1:-1]
				if tokens[6] == "values:":
					actual_val = float(tokens[7]) 
					predicted_val = float(tokens[10])
				else:
					actual_val = float(tokens[6]) 
					predicted_val = float(tokens[8])
				if actual_val != args.ignore_value:
					sequences[output_name].append( (actual_val, predicted_val) )

		except Exception as e:
			print line
			print list(enumerate(tokens))
			print e
			print traceback.format_exc()
			exit()

	return sequences

def discritize(l, args):
	outl = list()
	for tup in l:
		actual = tup[0]
		pred = tup[1]
		disc_actual = args.bin_width * round(actual / args.bin_width)
		disc_pred = args.bin_width * round(pred / args.bin_width)
		outl.append( (disc_actual, disc_pred) )
	return outl

def plot_confusion_matrix(cm, target_names, outfile, title='Confusion matrix', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(target_names))
	plt.xticks(tick_marks, target_names, rotation=90)
	plt.yticks(tick_marks, target_names)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(outfile)
	plt.clf()
	

def main(args):
	sequences = parse_log_file(args)

	try:
		os.makedirs(args.out_dir)
	except:
		pass

	for name in sequences:
		if name.endswith('loss'):
			sequences[name] = discritize(sequences[name], args)
		outfile = os.path.join(args.out_dir, "%s.png" % name)
		actual, pred = zip(*sequences[name])
		labels = list(set(actual + pred))
		labels.sort()
		labels = map(str, labels)
		actual = map(str, actual)
		pred = map(str, pred)
		cm = sklearn.metrics.confusion_matrix(actual, pred)
		if args.normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		plot_confusion_matrix(cm, labels, outfile)


def get_args():
	parser = argparse.ArgumentParser(description="Creates an LMDB of DocumentDatums")
	parser.add_argument('log_file', type=str,
						help='log file of caffe run')
	parser.add_argument('out_dir', type=str,
						help='out directory.  Created if it does not already exist')

	parser.add_argument('-b', '--bin-width', type=float, default=0.1,
						help='Bin width for discretization of continuous outputs')
	parser.add_argument('-i', '--ignore-value', type=float, default=-1,
						help='Ignore value placeholder')
	parser.add_argument('-n', '--normalize', action='store_true', default=False,
						help='Normalize rows of conf mats')


	args = parser.parse_args()
	return args
	

if __name__ == "__main__":
	args = get_args()
	main(args)
