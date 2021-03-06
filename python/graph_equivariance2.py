
import os
import sys
import ast
import collections
import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from utils import get_transforms, safe_mkdir

SMOOTH = 0.02

#SPLITS = ['train', 'test']
SPLITS = ['test']
#MODEL_TYPES = ['linear', 'mlp']
MODEL_TYPES = ['linear']
#LOSS_TYPES = ['l2', 'ce_soft', 'ce_hard']
LOSS_TYPES = ['l2']
#ALL_METRICS = ['norm_accuracy', 'accuracy', 'agreement', 'avg_jsd', 'avg_jss', 'avg_l2', 'avg_norm_l2', 'avg_norm_l2_sim']
ALL_METRICS = ['accuracy']
#ALL_METRICS = ['avg_norm_l2_sim']

# all these metrics are between 0 and 1 (well, mostly), where 1 is good and 0 is bad
NORM_METRICS = ['norm_accuracy', 'agreement', 'avg_jss', 'avg_norm_l2_sim']

def plot_lines(x_labels, title, line_dict, out_file):
	#plt.xlabel("Transforms")
	#plt.ylabel("Metrics")
	#title = " ".join(sorted(line_dict.items())[-1][0].split(" ")[:-1])
	plt.title(title, fontsize=32)
	plt.yticks(fontsize=22)
	plt.xticks(range(len(x_labels)), map(lambda s: s.upper(), x_labels), fontsize=20)
	#plt.xticks(range(len(x_labels)), x_labels, fontsize=20)
	idx = 0
	ls = [':', '-.', '-', '--']
	cs = ['r', 'black', 'b', 'g', 'orange']
	for name, vals in sorted(line_dict.items()):
		plt.plot(range(len(vals)), vals, ls[idx % len(ls)], color=cs[idx % len(cs)], label=name, linewidth=4)
		idx += 1

	plt.legend(loc='best', fontsize=18)
	#plt.legend(loc='lower left', fontsize=16)
	plt.savefig(out_file)
	plt.clf()


def plot_invariances(l_name, l_in_seq, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'invariance_plots')
	safe_mkdir(out_dir)
	for split in SPLITS:
		sdir = os.path.join(out_dir, split)
		safe_mkdir(sdir)
		for metric in ALL_METRICS:
			line_dict = {name: invariance_sequences[split][metric] for name, invariance_sequences in zip(l_name, l_in_seq)}
			out_file = os.path.join(sdir, metric + '.png')
			plot_lines(labels, "Invariance", line_dict, out_file)


def plot_compare(l_name, l_eq_seq, l_in_seq, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'compare_plots')
	safe_mkdir(out_dir)
	for model_type in MODEL_TYPES:
		for loss in LOSS_TYPES:
			for split in SPLITS:
				sdir = os.path.join(out_dir, model_type, loss, split)
				safe_mkdir(sdir)
				for metric in ALL_METRICS:
					line_dict = {"%s Inv" % name: invariance_sequences[split][metric] for name, invariance_sequences in zip(l_name, l_in_seq)}
					line_dict.update({"%s Equi" % name: equivariance_sequences[model_type][loss][split][metric] for name, equivariance_sequences in zip(l_name, l_eq_seq)})
					out_file = os.path.join(sdir, metric + '.png')
					#plot_lines(labels, "%s %s Equivariance compare" % (title_prefix, metric), line_dict, out_file)
					plot_lines(labels, title_prefix, line_dict, out_file)


# compute the elementwise reduction in error of l2 wrt l1
def compute_reduction(l1, l2, eps=SMOOTH):
	return [((1 - x1) - (1 - x2)) / (1 - x1 + SMOOTH) if x1 < (1 + SMOOTH) else 0 for x1, x2 in zip(l1, l2)]
	

def plot_reduction(l_name, l_eq_seq, l_in_seq, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'reduction_plots')
	safe_mkdir(out_dir)
	for model_type in MODEL_TYPES:
		for loss in LOSS_TYPES:
			for split in SPLITS:
				sdir = os.path.join(out_dir, model_type, loss, split)
				safe_mkdir(sdir)
				for metric in ALL_METRICS:
					line_dict = {name: compute_reduction(in_seq[split][metric], eq_seq[model_type][loss][split][metric]) 
						for name, in_seq, eq_seq in zip(l_name, l_in_seq, l_eq_seq)}
					out_file = os.path.join(sdir, metric + '.png')
					plot_lines(labels, "%s %s Reduction compare" % (title_prefix, metric), line_dict, out_file)


def plot_equivariances(l_name, l_eq_seq, out_dir, labels, title_prefix):
	out_dir = os.path.join(out_dir, 'equivariance_plots')
	safe_mkdir(out_dir)
	for model_type in MODEL_TYPES:
		for loss in LOSS_TYPES:
			for split in SPLITS:
				sdir = os.path.join(out_dir, model_type, loss, split)
				safe_mkdir(sdir)
				for metric in ALL_METRICS:
					line_dict = {name: equivariance_sequences[model_type][loss][split][metric] for name, equivariance_sequences in zip(l_name, l_eq_seq)}
					out_file = os.path.join(sdir, metric + '.png')
					plot_lines(labels, "Equivariance",  line_dict, out_file)


def reorder_center_transforms(transforms):
	first = transforms[0]
	del transforms[0]
	transforms.insert(len(transforms) / 2, first)
	return transforms


def reorder_transforms(transforms):
	center_transforms = ('rotation', 'shift', 'shear')
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
			labels.append("%.1f\n%.1f" % (float(tokens[1]), float(tokens[2])))
		elif transform.startswith('perspective'):
			labels.append(str(idx))
		elif transform.startswith('rotation'):
			labels.append(str(int(float(tokens[1]))))
		elif transform.startswith('crop'):
			labels.append("%d\n%d" % (idx / 5, idx % 5))
		elif transform.startswith('shear'):
			#labels.append(str(int(float(tokens[1]))) + " " + tokens[2])
			labels.append(str(int(float(tokens[1]))))
		else:
			labels.append(transform.split()[1][:3])
	return labels, transforms[-1].split()[0]


def main(out_dir, transform_file, result_dirs):
	safe_mkdir(out_dir)
	l_name, l_eq_seq, l_in_seq = list(), list(), list()
	transforms, _ = get_transforms(transform_file)
	transforms = reorder_transforms(transforms)
	for idx, in_dir in enumerate(result_dirs):
		#if idx == 0:
		#	name = "RVL-CDIP"
		#else:
		#	name = "ANDOC"
		if idx != 0:
			if 'rotation' in in_dir:
				name = "Rotation"
				name = "Rotation " + in_dir.split('/')[-4].split('_')[-1] 
			if 'color' in in_dir:
				name = "Color Jitter"
				name = "Color " + in_dir.split('/')[-4].split('_')[-1] 
			if 'crop' in in_dir:
				name = "Crop"
				name = "Crop " + in_dir.split('/')[-4].split('_')[-1] 
			if 'elastic' in in_dir:
				name = "Elastic Deformations"
				name = "Elastic " + in_dir.split('/')[-4].split('_')[-2] + " " + in_dir.split('/')[-4].split('_')[-1] 
			if 'noise' in in_dir:
				name = "Gaussian Noise"
				name = "Noise " + in_dir.split('/')[-4].split('_')[-1] 
			if 'blur' in in_dir:
				name = "Gaussian Blur"
				if "blur_1_5" in in_dir:
					name = "Blur 1.5"
				else:
					name = "Blur 3"
				#name = "Blur " + in_dir.split('/')[-4].split('_')[-1] 
			if 'mirror' in in_dir:
				name = "Mirror"
				name = "Mirror " + in_dir.split('/')[-4].split('_')[0].upper() 
			if 'perspective' in in_dir:
				name = "Perspective"
				name = "Perspective " + in_dir.split('/')[-4].split('_')[-1] 
			if 'shear' in in_dir:
				name = "Horizontal Shear"
				name = "Shear " + in_dir.split('/')[-4].split('_')[-1] 
				
		else:
			name = "Baseline"
		print name

		all_metrics = load_metrics(transforms, in_dir)
		label_names, title_prefix = format_labels(transforms)
		invariance_sequences = format_invariances(all_metrics)
		equivariance_sequences = format_equivariances(all_metrics)
		title_prefix = in_dir.split('/')[-4]

		l_name.append(name)
		l_eq_seq.append(equivariance_sequences)
		l_in_seq.append(invariance_sequences)

	plot_invariances(l_name, l_in_seq, out_dir, label_names, title_prefix)
	plot_equivariances(l_name, l_eq_seq, out_dir, label_names, title_prefix)
	plot_compare(l_name, l_eq_seq, l_in_seq, out_dir, label_names, title_prefix)
	#plot_reduction(l_name, l_eq_seq, l_in_seq, out_dir, label_names, title_prefix)
	

if __name__ == "__main__":
	out_dir = sys.argv[1]
	transform_file = sys.argv[2]
	result_dirs = sys.argv[3:]

	main(out_dir, transform_file, result_dirs)

