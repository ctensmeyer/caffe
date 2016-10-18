
import os
import sys
import ast
import collections
import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from utils import get_transforms, safe_mkdir

SPLITS = ['train', 'test']
MODEL_TYPES = ['linear', 'mlp']
LOSS_TYPES = ['l2', 'ce_soft', 'ce_hard']
METRICS = ['accuracy', 'agreement', 'avg_jsd', 'avg_l2']

def plot_lines(x_labels, title, line_dict, out_file):
	plt.xlabel("Transforms")
	plt.ylabel("Metrics")
	plt.title(title)
	plt.xticks(range(len(x_labels)), x_labels)
	for name, vals in line_dict.items():
		plt.plot(range(len(vals)), vals, label=name)

	plt.legend(loc='best')
	plt.savefig(out_file)
	plt.clf()


def plot_invariances(invariance_sequences, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'invariance_plots')
	safe_mkdir(out_dir)
	for split in SPLITS:
		sdir = os.path.join(out_dir, split)
		safe_mkdir(sdir)
		for metric in METRICS:
			line_dict = {metric: invariance_sequences[split][metric]}
			out_file = os.path.join(sdir, metric + '.png')
			plot_lines(labels, "%s %s Invariance" % (title_prefix, metric), line_dict, out_file)

		line_dict = {metric: invariance_sequences[split][metric] for metric in METRICS}
		out_file = os.path.join(sdir, 'all.png')
		plot_lines(labels, "%s All Metrics Invariance" % title_prefix, line_dict, out_file)


def plot_equivariances(equivariance_sequences, invariance_sequences, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'equivariance_plots')
	safe_mkdir(out_dir)
	for model_type in MODEL_TYPES:
		for loss in LOSS_TYPES:
			for split in SPLITS:
				sdir = os.path.join(out_dir, model_type, loss, split)
				safe_mkdir(sdir)
				for metric in METRICS:
					line_dict = {metric: equivariance_sequences[model_type][loss][split][metric],
								 "invariance_%s" % metric: invariance_sequences[split][metric]}
					out_file = os.path.join(sdir, metric + '.png')
					plot_lines(labels, "%s %s Equivariance" % (title_prefix, metric), line_dict, out_file)

				line_dict = {metric: equivariance_sequences[model_type][loss][split][metric] for metric in METRICS}
				line_dict.update({"invariance_%s" % metric: invariance_sequences[split][metric] for metric in METRICS})
				out_file = os.path.join(sdir, 'all.png')
				plot_lines(labels, "%s All Metrics Equivariance" % title_prefix, line_dict, out_file)


def plot_loss_equivariance_compare(equivariance_sequences, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'loss_compare_equivariance_plots')
	safe_mkdir(out_dir)
	for model_type in MODEL_TYPES:
		for split in SPLITS:
			sdir = os.path.join(out_dir, model_type, split)
			safe_mkdir(sdir)
			for metric in METRICS:
				line_dict = {"%s_%s" % (metric, loss): equivariance_sequences[model_type][loss][split][metric] for loss in LOSS_TYPES}
				out_file = os.path.join(sdir, metric + '.png')
				plot_lines(labels, "%s %s Equivariance Loss Compare" % (title_prefix, metric), line_dict, out_file)


def plot_split_equivariance_compare(equivariance_sequences, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'split_compare_equivariance_plots')
	safe_mkdir(out_dir)
	for model_type in MODEL_TYPES:
		for loss in LOSS_TYPES:
			sdir = os.path.join(out_dir, model_type, loss)
			safe_mkdir(sdir)
			for metric in METRICS:
				line_dict = {"%s_%s" % (metric, split): equivariance_sequences[model_type][loss][split][metric] for split in SPLITS}
				out_file = os.path.join(sdir, metric + '.png')
				plot_lines(labels, "%s %s Equivariance Split Compare" % (title_prefix, metric), line_dict, out_file)


def plot_model_equivariance_compare(equivariance_sequences, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'model_compare_equivariance_plots')
	safe_mkdir(out_dir)
	for loss in LOSS_TYPES:
		for split in SPLITS:
			sdir = os.path.join(out_dir, loss, split)
			safe_mkdir(sdir)
			for metric in METRICS:
				line_dict = {"%s_%s" % (metric, model_type): equivariance_sequences[model_type][loss][split][metric] for model_type in MODEL_TYPES}
				out_file = os.path.join(sdir, metric + '.png')
				plot_lines(labels, "%s %s Equivariance Model Compare" % (title_prefix, metric), line_dict, out_file)


def reorder_rotation_transforms(transforms):
	first = transforms[0]
	del transforms[0]
	transforms.insert(len(transforms) / 2, first)
	return transforms


def reorder_transforms(transforms):
	if transforms[-1].startswith("rotation"):
		return reorder_rotation_transforms(transforms)
	#elif transforms[-1].startswith("shear"):
	#	return reorder_rotation_transforms(transforms)
	else:
		return transforms

def load_metrics(transforms, in_dir):
	all_metrics = list()
	for transform in transforms:
		file_path = os.path.join(in_dir, transform.replace(' ', '_') + '.txt')
		if not os.path.exists(file_path):
			raise Exception("File %s does not exist" % file_path)
		file_contents = open(file_path, 'r').read()
		metrics = ast.literal_eval(file_contents)
		all_metrics.append( (transform, metrics) )
	return all_metrics


# train_v_test/metric
def format_invariances(all_metrics):
	invariances = {split: collections.defaultdict(list) for split in SPLITS}
	for transform, metrics in all_metrics:
		for split in SPLITS:
			for metric in METRICS:
				invariances[split][metric].append(metrics[split]['invariance'][metric])
	return invariances
	

# model_type/loss/train_v_test/metric
def format_equivariances(all_metrics):
	equivariances = dict()
	# set up structure
	for model_type in MODEL_TYPES:
		equivariances[model_type] = dict()
		for loss in LOSS_TYPES:
			equivariances[model_type][loss] = dict()
			for split in SPLITS:
				equivariances[model_type][loss][split] = dict()
				for metric in METRICS:
					equivariances[model_type][loss][split][metric] = list()
	for transform, metrics in all_metrics:
			for model_type in MODEL_TYPES:
				for loss in LOSS_TYPES:
					for split in SPLITS:
						for metric in METRICS:
							equivariances[model_type][loss][split][metric].append(metrics[split]['equivariance'][model_type][loss].get(metric, 0))
	return equivariances


def format_labels(transforms):
	labels = list()
	for transform in transforms:
		if transform == 'none' or transform == 'resize 227 277':
			labels.append('baseline')
		else:
			labels.append(transform.split()[1])
	return labels, transforms[-1].split()[0]


def main(transform_file, in_dir, out_dir):
	safe_mkdir(out_dir)
	transforms, _ = get_transforms(transform_file)
	transforms = reorder_transforms(transforms)
	all_metrics = load_metrics(transforms, in_dir)
	label_names, title_prefix = format_labels(transforms)
	invariance_sequences = format_invariances(all_metrics)
	#pprint.pprint(invariance_sequences)
	equivariance_sequences = format_equivariances(all_metrics)
	#print
	#pprint.pprint(equivariance_sequences)

	plot_invariances(invariance_sequences, out_dir, label_names, title_prefix)
	plot_equivariances(equivariance_sequences, invariance_sequences, out_dir, label_names, title_prefix)
	plot_loss_equivariance_compare(equivariance_sequences, out_dir, label_names, title_prefix)
	plot_model_equivariance_compare(equivariance_sequences, out_dir, label_names, title_prefix)
	plot_split_equivariance_compare(equivariance_sequences, out_dir, label_names, title_prefix)

	

if __name__ == "__main__":
	transform_file = sys.argv[1]
	in_dir = sys.argv[2]
	out_dir = sys.argv[3]

	main(transform_file, in_dir, out_dir)

