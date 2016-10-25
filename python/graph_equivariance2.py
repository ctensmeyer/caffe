
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
#MODEL_TYPES = ['linear', 'mlp']
MODEL_TYPES = ['linear']
#LOSS_TYPES = ['l2', 'ce_soft', 'ce_hard']
LOSS_TYPES = ['l2']
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


def plot_invariances(l_name, l_in_seq, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'invariance_plots')
	safe_mkdir(out_dir)
	for split in SPLITS:
		sdir = os.path.join(out_dir, split)
		safe_mkdir(sdir)
		for metric in METRICS:
			line_dict = {name: invariance_sequences[split][metric] for name, invariance_sequences in zip(l_name, l_in_seq)}
			out_file = os.path.join(sdir, metric + '.png')
			plot_lines(labels, "%s %s Invariance" % (title_prefix, metric), line_dict, out_file)


def plot_compare(l_name, l_eq_seq, l_in_seq, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'compare_plots')
	safe_mkdir(out_dir)
	for model_type in MODEL_TYPES:
		for loss in LOSS_TYPES:
			for split in SPLITS:
				sdir = os.path.join(out_dir, model_type, loss, split)
				safe_mkdir(sdir)
				for metric in METRICS:
					line_dict = {"in_%s" % name: invariance_sequences[split][metric] for name, invariance_sequences in zip(l_name, l_in_seq)}
					line_dict.update({"eq_%s" % name: equivariance_sequences[model_type][loss][split][metric] for name, equivariance_sequences in zip(l_name, l_eq_seq)})
					out_file = os.path.join(sdir, metric + '.png')
					plot_lines(labels, "%s %s Equivariance compare" % (title_prefix, metric), line_dict, out_file)


def plot_equivariances(l_name, l_eq_seq, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'equivariance_plots')
	safe_mkdir(out_dir)
	for model_type in MODEL_TYPES:
		for loss in LOSS_TYPES:
			for split in SPLITS:
				sdir = os.path.join(out_dir, model_type, loss, split)
				safe_mkdir(sdir)
				for metric in METRICS:
					line_dict = {name: equivariance_sequences[model_type][loss][split][metric] for name, equivariance_sequences in zip(l_name, l_eq_seq)}
					out_file = os.path.join(sdir, metric + '.png')
					plot_lines(labels, "%s %s Equivariance" % (title_prefix, metric), line_dict, out_file)



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


def reorder_center_transforms(transforms):
	first = transforms[0]
	del transforms[0]
	transforms.insert(len(transforms) / 2, first)
	return transforms


def reorder_transforms(transforms):
	center_transforms = ('rotation', 'crop', 'shift')
	if transforms[-1].startswith(center_transforms):
		return reorder_center_transforms(transforms)
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
							equivariances[model_type][loss][split][metric].append(metrics[split]['equivariance'][model_type][loss].get(metric,
								metrics[split]['invariance'][metric]))
	return equivariances


def format_labels(transforms):
	labels = list()
	for idx, transform in enumerate(transforms):
		tokens = transform.split()
		if transform == 'none':
			labels.append('0')
		elif transform.startswith('elastic'):
			labels.append(".1f%/.1f%" % (tokens[1], tokens[2]))
		elif transform.startswith('perspective'):
			labels.append(str(idx))
		elif transform.startswith('rotation'):
			labels.append(str(int(tokens[1])))
		elif transform.startswith('shear'):
			labels.append(str(int(tokens[1])) + " " + tokens[2])
		else:
			labels.append(transform.split()[1][:3])
	return labels, transforms[-1].split()[0]


def main(out_dir, net_dirs):
	safe_mkdir(out_dir)
	l_name, l_eq_seq, l_in_seq = list(), list(), list()
	for net_dir in net_dirs:
		in_dir = os.path.join(net_dir, 'equivariance', 'results')
		transform_file = os.path.join(net_dir, 'equivariance_transforms.txt')
		name = net_dir.split('/')[-2]

		transforms, _ = get_transforms(transform_file)
		transforms = reorder_transforms(transforms)
		all_metrics = load_metrics(transforms, in_dir)
		label_names, title_prefix = format_labels(transforms)
		invariance_sequences = format_invariances(all_metrics)
		equivariance_sequences = format_equivariances(all_metrics)

		l_name.append(name)
		l_eq_seq.append(equivariance_sequences)
		l_in_seq.append(invariance_sequences)

	plot_invariances(l_name, l_in_seq, out_dir, label_names, title_prefix)
	plot_equivariances(l_name, l_eq_seq, out_dir, label_names, title_prefix)
	plot_compare(l_name, l_eq_seq, l_in_seq, out_dir, label_names, title_prefix)
	

if __name__ == "__main__":
	out_dir = sys.argv[1]
	net_dirs = sys.argv[2:]

	main(out_dir, net_dirs)

