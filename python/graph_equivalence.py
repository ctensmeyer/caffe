
import os
import sys
import ast
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from utils import safe_mkdir
import sklearn.manifold

matplotlib.rcParams.update({'font.size': 26})

np.set_printoptions(threshold=3000, linewidth=185)

#METRICS = [('accuracy', False),
#           ('agreement', False),
#		   ('avg_jsd', True),
#		   ('avg_l2', True),
#		   ('avg_scaled_l2', True)]

METRICS = [ ('avg_scaled_l2', True)]
MODEL_TYPES = ['linear']
LOSS_TYPES = ['l2']
#SPLITS = ['train', 'test']
SPLITS = ['test']

SCALE = 100

color_names = {
'black': 'baseline',
'red': 'color_jitter',
'blue': 'crop',
'green': 'elastic',
'orange': 'blur',
'purple': 'noise',
'brown': 'mirror',
'cyan': 'perspective',
'olive': 'rotation',
'darkgray': 'shear'
}

color_sizes = {
'baseline' : ('black', 30, 'o'),

'color_jitter_5' : ('red', 55, 'v'),
'color_jitter_10' : ('red', 70, 'v'),
'color_jitter_15' : ('red', 85, 'v'),
'color_jitter_20' : ('red', 100, 'v'),

'crop_240' : ('blue', 55, '^'),
'crop_256' : ('blue', 70, '^'),
'crop_288' : ('blue', 85, '^'),
'crop_320' : ('blue', 100, '^'),

'elastic_2_5' : ('green', 55, '<'),
'elastic_2_10' : ('green', 70, '<'),
'elastic_3_5' : ('green', 85, '<'),
'elastic_3_10' : ('green', 100, '<'),

'gauss_blur_1_5' : ('orange', 70, '>'),
'gauss_blur_3' : ('orange', 85, '>'),

'gauss_noise_5' : ('purple', 55, 'x'),
'gauss_noise_10' : ('purple', 70, 'x'),
'gauss_noise_15' : ('purple', 85, 'x'),
'gauss_noise_20' : ('purple', 100, 'x'),

'h_mirror' : ('brown', 55, 's'),
'hv_mirror' : ('brown', 70, 's'),
'v_mirror' : ('brown', 85, 's'),

'perspective_1' : ('cyan', 55, '*'),
'perspective_2' : ('cyan', 70, '*'),
'perspective_3' : ('cyan', 85, '*'),
'perspective_4' : ('cyan', 100, '*'),

'rotation_5' : ('olive', 55, 'H'),
'rotation_10' : ('olive', 70, 'H'),
'rotation_15' : ('olive', 85, 'H'),
'rotation_20' : ('olive', 100, 'H'),

'shear_5' : ('darkgray', 55, 'D'),
'shear_10' : ('darkgray', 70, 'D'),
'shear_15' : ('darkgray', 85, 'D'),
'shear_20' : ('darkgray', 100, 'D'),
}

def num_knns(dist_mat, groups):
	total = float(np.sum(groups, axis=None)) - np.sum(groups[np.diag_indices(dist_mat.shape[0])], axis=None)
	sorted = np.argsort(dist_mat, axis=1)
	num_nebs = [0]
	for k in xrange(1, dist_mat.shape[0]):
		nebs = 0

		# iterate over rows
		for x in xrange(dist_mat.shape[0]):

			# iterate over the nearest neighbors
			y = 0
			nebs_seen = 0
			while nebs_seen < k:
				if x == sorted[x,y]:
					#skip the self
					y += 1

				if groups[x, sorted[x,y]]:
					nebs += 1
				y += 1
				nebs_seen += 1
		num_nebs.append(nebs / total)
	return num_nebs

def graph_embedding(out_file, Y, colors, sizes, markers):
	d = dict()
	for idx in xrange(Y.shape[0]):
		tmp = plt.scatter([Y[idx,0]], [Y[idx,1]], c=[colors[idx]], s=[sizes[idx]], marker=markers[idx])
		d[markers[idx]] = tmp
	#plt.legend( (d['o'], d['v'], d['^'], d['<'], d['>'], d['x'], d['s'], d['*'], d['H'], d['D']),
	#		('Baseline', 'Color Jitter', 'Crop', 'Elastic', 'Blur', 'Noise', 
	#			'Mirror', 'Perspective', 'Rotation', 'Shear'),
	#		loc='lower left',
	#		scatterpoints=1,
	#		ncol=5,
	#		fontsize=16)
	plt.xticks([], [])
	plt.yticks([], [])
	plt.savefig(out_file)
	plt.clf()

def graph_nebs(out_file, nebs, rnebs):
	plt.plot(nebs, label="Measured Distances", linewidth=3)
	plt.plot(rnebs, '--', label="Random Distances", linewidth=3)
	plt.xlabel("K")
	plt.ylabel("Fraction of Pairs")
	plt.title("Number of Same-Tranform Pairs\nwithin K Neighbors")
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig(out_file)
	plt.clf()

cdip=[0, 0.2857142857142857, 0.4387755102040816, 0.5612244897959183, 0.6632653061224489, 0.7244897959183674, 0.8061224489795918, 0.8469387755102041, 0.8673469387755102, 0.8775510204081632, 0.8877551020408163, 0.9081632653061225, 0.9081632653061225, 0.9081632653061225, 0.9183673469387755, 0.9285714285714286, 0.9285714285714286, 0.9489795918367347, 0.9693877551020408, 0.9693877551020408, 0.9693877551020408, 0.9693877551020408, 0.9795918367346939, 0.9795918367346939, 0.9795918367346939, 0.9795918367346939, 0.9795918367346939, 0.9795918367346939, 0.9795918367346939, 0.9795918367346939, 0.9795918367346939, 0.9897959183673469, 0.9897959183673469, 0.9897959183673469, 1.0, 1.0]
imagenet=[0, 0.10204081632653061, 0.14285714285714285, 0.25510204081632654, 0.336734693877551, 0.3979591836734694, 0.45918367346938777, 0.5408163265306123, 0.6122448979591837, 0.6938775510204082, 0.7755102040816326, 0.826530612244898, 0.8775510204081632, 0.9183673469387755, 0.9285714285714286, 0.9285714285714286, 0.9489795918367347, 0.9489795918367347, 0.9591836734693877, 0.9591836734693877, 0.9591836734693877, 0.9693877551020408, 0.9795918367346939, 0.9795918367346939, 0.9795918367346939, 0.9795918367346939, 0.9897959183673469, 0.9897959183673469, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

def graph_nebs2(out_file, cdip, imagenet, rnebs):
	matplotlib.rcParams.update({'font.size': 24})
	plt.figure(figsize=(10, 12))
	plt.plot(cdip, label="RVL-CDIP", linewidth=3)
	plt.plot(imagenet, '--', label="ILSVRC", linewidth=3)
	plt.plot(rnebs, '-.', label="Random Distances", linewidth=3)
	plt.xlabel("K", fontsize=26)
	plt.ylabel("Fraction of Pairs", fontsize=26)
	plt.title("Number of Same-Tranform Pairs\nwithin K Neighbors", fontsize=34)
	plt.legend(loc='best', fontsize=26)
	plt.tight_layout()
	plt.savefig(out_file)
	plt.clf()

def graph_heatmap(out_file, dists, labels):
	#dists[np.diag_indices(dists.shape[0])] = dists.max()
	plt.figure(figsize=(12,12))
	plt.imshow(dists, interpolation='nearest', cmap=plt.cm.bone)
	plt.colorbar()
	plt.title("Transform Distances")
	tick_marks = np.arange(len(labels))
	plt.xticks(tick_marks, labels, rotation=90, fontsize=18)
	plt.yticks(tick_marks, labels, fontsize=18)
	plt.tight_layout()
	plt.savefig(out_file)
	plt.clf()
	

def get_color_sizes(net_dirs):
	colors, sizes, markers = list(), list(), list()
	for net_dir in net_dirs:
		aug = net_dir.split('/')[-2]
		color, size, marker = color_sizes[aug]
		colors.append(color)
		sizes.append(size)
		markers.append(marker)
	return colors, sizes, markers

def get_random_knn(groups):
	#all_rnebs = list()
	#for x in xrange(100):
	#	dist_mat = np.random.random( (num_net_dirs, num_net_dirs) )
	#	sym_dist_mat = (dist_mat + dist_mat.T) / 2
	#	nebs = num_knns(sym_dist_mat, groups)
	#	all_rnebs.append(nebs)

	#rnebs = np.average(np.asarray(all_rnebs), axis=0)
	#return rnebs
	return [float(k) / (groups.shape[0] - 1) for k in xrange(groups.shape[0])]
				

ROOT='/fslhome/waldol1/fsl_groups/fslg_nnml/compute/experiments/preprocessing/nets'
group_file = sys.argv[1]
root_out_dir = sys.argv[2]

net_dirs = open(group_file, 'r').readlines()
net_dirs = map(lambda s: s.rstrip(), net_dirs)
num_net_dirs = len(net_dirs)

colors, sizes, markers = get_color_sizes(net_dirs)
groups = np.asarray([ [c1 == c2 for c2 in colors] for c1 in colors])

tsne = sklearn.manifold.TSNE(n_components=2, method='exact', metric='precomputed', n_iter=10000, n_iter_without_progress=100)
rnebs = get_random_knn(groups)
#graph_nebs2('out.png', cdip, imagenet, rnebs)
#exit()

def get_labels(net_dirs):
	labels = list()
	for net_dir in net_dirs:
		if "baseline" in net_dir:
			labels.append("Ba")
		elif "blur" in net_dir:
			labels.append("Bl")
		elif "color" in net_dir:
			labels.append("Co")
		elif "crop" in net_dir:
			labels.append("Cr")
		elif "elastic" in net_dir:
			labels.append("E")
		elif "noise" in net_dir:
			labels.append("N")
		elif "perspective" in net_dir:
			labels.append("P")
		elif "rotation" in net_dir:
			labels.append("R")
		elif "shear" in net_dir:
			labels.append("S")
		elif "mirror" in net_dir:
			labels.append("M")
		else:
			labels.append("")

	return labels


for split in SPLITS:
	print "Starting Split:", split
	for loss in LOSS_TYPES:
		for model_type in MODEL_TYPES:
			#out_dir = os.path.join(root_out_dir, split, loss, model_type)
			out_dir = os.path.join(root_out_dir, split)
			safe_mkdir(out_dir)
			for metric, is_distance in METRICS:
				print "Starting Metric:", metric

				dist_mat = np.zeros( (num_net_dirs, num_net_dirs), dtype=float)
				for idx1, net_dir1 in enumerate(net_dirs):
					result_dir = os.path.join(ROOT, net_dir1, 'equivalence/results')
					for idx2, net_dir2 in enumerate(net_dirs):
						fn = net_dir2.replace('/', '_') + '.txt'
						result_file = os.path.join(result_dir, fn)
						results = ast.literal_eval(open(result_file, 'r').read())
						if is_distance:
							dist_mat[idx1,idx2] = SCALE * results[split][model_type][loss][metric]
						else:
							dist_mat[idx1,idx2] = SCALE * (1 - results[split][model_type][loss][metric])

				np.savetxt(os.path.join(out_dir, "raw_dist_%s.txt" % metric), dist_mat, fmt='%.2f')
				sym_dist_mat = (dist_mat + dist_mat.T) / 2
				np.savetxt(os.path.join(out_dir, "sym_dist_%s.txt" % metric), sym_dist_mat, fmt='%.2f')
				diff_mat = np.abs(dist_mat - dist_mat.T)
				np.savetxt(os.path.join(out_dir, "diff_%s.txt" % metric), diff_mat, fmt='%6.3f')
				out_file = os.path.join(out_dir, "diff_heatmap_%s.png" % metric)
				graph_heatmap(out_file, diff_mat, get_labels(net_dirs))
				exit()


				#for x in xrange(20):
				#	embedding = tsne.fit_transform(sym_dist_mat)
				#	out_im = os.path.join(out_dir, "embedding_%s_%d.png" % (metric, x))
				#	graph_embedding(out_im, embedding, colors, sizes, markers)

				nebs = num_knns(sym_dist_mat, groups)
				out_im = os.path.join(out_dir, "nebs_all_%s.png" % metric)
				graph_nebs(out_im, nebs, rnebs)

				# knns by group
				all_colors = list(set(colors))
				all_colors.sort()
				sub_out_dir = os.path.join(out_dir, "group_nebs_%s" % metric)
				safe_mkdir(sub_out_dir)
				print "Knns by group"
				for color in all_colors:
					continue
					sub_group = np.asarray([ [c1 == c2 and c1 == color for c2 in colors] for c1 in colors])

					nebs = num_knns(sym_dist_mat, sub_group)
					rnebs = get_random_knn(sub_group)
					out_im = os.path.join(sub_out_dir, "%s_%s.png" % (color_names[color], metric))
					graph_nebs(out_im, nebs, rnebs)

				# pairwise embeddings 
				sub_out_dir = os.path.join(out_dir, "pairwise_%s" % metric)
				group_distances = np.zeros((10,10))
				safe_mkdir(sub_out_dir)
				group_order = list()
				print "Pairwise"
				for idx1, color1 in enumerate(all_colors):
					print "Pairwise", idx1
					group_order.append(color_names[color1])
					for idx2, color2 in enumerate(all_colors):
						if idx1 >= idx2:
							continue
						sub_groups = np.asarray([ [c1 == c2 and (c1 == color1 or c1 == color2) for c2 in colors] for c1 in colors])
						indices = []
						indices_1 = []
						indices_2 = []
						sub_colors = []
						sub_sizes = []
						sub_markers = []
						idx = 0
						for c,s,m in zip(colors, sizes, markers):
							if c == color1 or c == color2:
								indices.append(idx)
								sub_colors.append(c)
								sub_sizes.append(s)
								sub_markers.append(m)
							if c == color1:
								indices_1.append(idx)
							if c == color2:
								indices_2.append(idx)
							idx += 1
						sub_sym_dist_mat = sym_dist_mat[indices, :]
						sub_sym_dist_mat = sub_sym_dist_mat[:, indices]

						#embedding = tsne.fit_transform(sub_sym_dist_mat)
						#out_im = os.path.join(sub_out_dir, "embedding_%s_%s.png" % (color_names[color1], color_names[color2]))
						#graph_embedding(out_im, embedding, sub_colors, sub_sizes, sub_markers)

						avg_inter_group_dist = np.average(sym_dist_mat[indices_1,:][:, indices_2], axis=None) / np.max(sym_dist_mat, axis=None)
						group_distances[idx1,idx2] = avg_inter_group_dist
						group_distances[idx2,idx1] = avg_inter_group_dist
				np.savetxt(os.path.join(out_dir, "group_dists_%s.txt" % metric), group_distances, fmt='%6.3f')
				out_file = os.path.join(out_dir, "group_dists_%s.png" % metric)
				graph_heatmap(out_file, group_distances, group_order)

				open(os.path.join(out_dir, "group_order.txt"), 'w').write("\n".join(group_order))


