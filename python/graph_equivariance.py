
import os
import sys
import ast
import collections
import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from utils import get_transforms, safe_mkdir


ERROR_ON_NOT_FOUND = False
SMOOTH = 0.02

SPLITS = ['train', 'test']
#MODEL_TYPES = ['linear', 'mlp']
MODEL_TYPES = ['linear']
#LOSS_TYPES = ['l2', 'ce_soft', 'ce_hard']
LOSS_TYPES = ['l2']
ALL_METRICS = ['norm_accuracy', 'accuracy', 'agreement', 'avg_jsd', 'avg_jss', 'avg_l2', 'avg_norm_l2', 'avg_norm_l2_sim']

# all these metrics are between 0 and 1 (well, mostly), where 1 is good and 0 is bad
NORM_METRICS = ['norm_accuracy', 'agreement', 'avg_jss', 'avg_norm_l2_sim']

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
		for metric in ALL_METRICS:
			line_dict = {metric: invariance_sequences[split][metric]}
			out_file = os.path.join(sdir, metric + '.png')
			plot_lines(labels, "%s %s Invariance" % (title_prefix, metric), line_dict, out_file)


def plot_equivariances(equivariance_sequences, invariance_sequences, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'equivariance_plots')
	safe_mkdir(out_dir)
	for model_type in MODEL_TYPES:
		for loss in LOSS_TYPES:
			for split in SPLITS:
				sdir = os.path.join(out_dir, model_type, loss, split)
				safe_mkdir(sdir)
				for metric in ALL_METRICS:
					line_dict = {metric: equivariance_sequences[model_type][loss][split][metric],
								 "invariance_%s" % metric: invariance_sequences[split][metric]}
					out_file = os.path.join(sdir, metric + '.png')
					plot_lines(labels, "%s %s Equivariance" % (title_prefix, metric), line_dict, out_file)


# compute the elementwise reduction in error of l2 wrt l1
def compute_reduction(l1, l2, eps=SMOOTH):
	return [((1 - x1) - (1 - x2)) / (1 - x1 + SMOOTH) if x1 < (1 + SMOOTH) else 0 for x1, x2 in zip(l1, l2)]


def plot_reductions(equivariance_sequences, invariance_sequences, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'reduction_plots')
	safe_mkdir(out_dir)
	for model_type in MODEL_TYPES:
		for loss in LOSS_TYPES:
			for split in SPLITS:
				sdir = os.path.join(out_dir, model_type, loss, split)
				safe_mkdir(sdir)
				for metric in NORM_METRICS:
					eq_seq = equivariance_sequences[model_type][loss][split][metric]
					in_seq = invariance_sequences[split][metric]

					# compute relative reduction in metric error
					red_seq = compute_reduction(in_seq, eq_seq)
					line_dict = {metric: red_seq}
					out_file = os.path.join(sdir, metric + '.png')
					plot_lines(labels, "%s %s Equivariance" % (title_prefix, metric), line_dict, out_file)


def plot_loss_equivariance_compare(equivariance_sequences, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'loss_compare_equivariance_plots')
	safe_mkdir(out_dir)
	for model_type in MODEL_TYPES:
		for split in SPLITS:
			sdir = os.path.join(out_dir, model_type, split)
			safe_mkdir(sdir)
			for metric in ALL_METRICS:
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
			for metric in ALL_METRICS:
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
			for metric in ALL_METRICS:
				line_dict = {"%s_%s" % (metric, model_type): equivariance_sequences[model_type][loss][split][metric] for model_type in MODEL_TYPES}
				out_file = os.path.join(sdir, metric + '.png')
				plot_lines(labels, "%s %s Equivariance Model Compare" % (title_prefix, metric), line_dict, out_file)


def reorder_center_transforms(transforms):
	first = transforms[0]
	del transforms[0]
	transforms.insert(len(transforms) / 2, first)
	return transforms


def reorder_transforms(transforms):
	center_transforms = ('rotation', 'crop', 'shift', 'shear')
	if transforms[-1].startswith(center_transforms):
		return reorder_center_transforms(transforms)
	#elif transforms[-1].startswith("shear"):
	#	return reorder_rotation_transforms(transforms)
	else:
		return transforms

def load_metrics(transforms, in_dir):
	all_metrics = list()
	included_transforms = list()
	for transform in transforms:
		file_path = os.path.join(in_dir, transform.replace(' ', '_') + '.txt')
		if not os.path.exists(file_path):
			if ERROR_ON_NOT_FOUND:
				raise Exception("File %s does not exist" % file_path)
			else:
				print "File %s does not exist" % file_path
				continue
		file_contents = open(file_path, 'r').read()
		metrics = ast.literal_eval(file_contents)
		all_metrics.append( (transform, metrics) )
		included_transforms.append(transform)
	return all_metrics, included_transforms


# train_v_test/metric
def format_invariances(all_metrics):
	invariances = {split: collections.defaultdict(list) for split in SPLITS}
	for transform, metrics in all_metrics:
		for split in SPLITS:
			for metric in ALL_METRICS:
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
				for metric in ALL_METRICS:
					equivariances[model_type][loss][split][metric] = list()
	for transform, metrics in all_metrics:
			for model_type in MODEL_TYPES:
				for loss in LOSS_TYPES:
					for split in SPLITS:
						for metric in ALL_METRICS:
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
			labels.append("%.1f/%.1f" % (float(tokens[1]), float(tokens[2])))
		elif transform.startswith('perspective'):
			labels.append(str(idx))
		elif transform.startswith('rotation'):
			labels.append(str(int(float(tokens[1]))))
		elif transform.startswith('shear'):
			labels.append(str(int(float(tokens[1]))) + " " + tokens[2])
		else:
			labels.append(transform.split()[1][:3])
	return labels, transforms[-1].split()[0]


def main(transform_file, in_dir, out_dir):
	safe_mkdir(out_dir)
	transforms, _ = get_transforms(transform_file)
	transforms = reorder_transforms(transforms)
	all_metrics, transforms = load_metrics(transforms, in_dir)
	label_names, title_prefix = format_labels(transforms)
	invariance_sequences = format_invariances(all_metrics)
	equivariance_sequences = format_equivariances(all_metrics)

	plot_invariances(invariance_sequences, out_dir, label_names, title_prefix)
	plot_equivariances(equivariance_sequences, invariance_sequences, out_dir, label_names, title_prefix)
	plot_reductions(equivariance_sequences, invariance_sequences, out_dir, label_names, title_prefix)
	#plot_loss_equivariance_compare(equivariance_sequences, out_dir, label_names, title_prefix)
	#plot_model_equivariance_compare(equivariance_sequences, out_dir, label_names, title_prefix)
	plot_split_equivariance_compare(equivariance_sequences, out_dir, label_names, title_prefix)

	

if __name__ == "__main__":
	transform_file = sys.argv[1]
	in_dir = sys.argv[2]
	out_dir = sys.argv[3]

	main(transform_file, in_dir, out_dir)

