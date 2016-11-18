
import os
import sys
import ast
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from utils import safe_mkdir
import sklearn.manifold

np.set_printoptions(threshold=3000, linewidth=150)

METRICS = [('accuracy', False),
           ('agreement', False),
		   ('avg_jsd', True),
		   ('avg_l2', True),
		   ('avg_scaled_l2', True)]
MODEL_TYPES = ['linear']
LOSS_TYPES = ['l2']
SPLITS = ['train', 'test']

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
'baseline' : ('black', 30),

'color_jitter_5' : ('red', 20),
'color_jitter_10' : ('red', 35),
'color_jitter_15' : ('red', 50),
'color_jitter_20' : ('red', 65),

'crop_240' : ('blue', 20),
'crop_256' : ('blue', 35),
'crop_288' : ('blue', 50),
'crop_320' : ('blue', 65),

'elastic_2_5' : ('green', 20),
'elastic_2_10' : ('green', 35),
'elastic_3_5' : ('green', 50),
'elastic_3_10' : ('green', 65),

'gauss_blur_1_5' : ('orange', 35),
'gauss_blur_3' : ('orange', 50),

'gauss_noise_5' : ('purple', 20),
'gauss_noise_10' : ('purple', 35),
'gauss_noise_15' : ('purple', 50),
'gauss_noise_20' : ('purple', 65),

'h_mirror' : ('brown', 20),
'hv_mirror' : ('brown', 35),
'v_mirror' : ('brown', 50),

'perspective_1' : ('cyan', 20),
'perspective_2' : ('cyan', 35),
'perspective_3' : ('cyan', 50),
'perspective_4' : ('cyan', 65),

'rotation_5' : ('olive', 20),
'rotation_10' : ('olive', 35),
'rotation_15' : ('olive', 50),
'rotation_20' : ('olive', 65),

'shear_5' : ('darkgray', 20),
'shear_10' : ('darkgray', 35),
'shear_15' : ('darkgray', 50),
'shear_20' : ('darkgray', 65),
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

def graph_embedding(out_file, Y, colors, sizes):
	plt.scatter(Y[:,0], Y[:,1], c=colors, s=sizes)
	plt.savefig(out_file)
	plt.clf()

def graph_nebs(out_file, nebs, rnebs):
	plt.plot(nebs, label="Actual NNs")
	plt.plot(rnebs, label="Random NNs")
	plt.xlabel("K")
	plt.ylabel("Total K-NN within groups")
	plt.title("Number of KNN Pairs of the same transform")
	plt.legend(loc='best')
	plt.savefig(out_file)
	plt.clf()

def get_color_sizes(net_dirs):
	colors, sizes = list(), list()
	for net_dir in net_dirs:
		aug = net_dir.split('/')[-2]
		color, size = color_sizes[aug]
		colors.append(color)
		sizes.append(size)
	return colors, sizes

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

colors, sizes = get_color_sizes(net_dirs)
groups = np.asarray([ [c1 == c2 for c2 in colors] for c1 in colors])

tsne = sklearn.manifold.TSNE(n_components=2, method='exact', metric='precomputed')
rnebs = get_random_knn(groups)


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
				diff_mat = dist_mat - dist_mat.T
				np.savetxt(os.path.join(out_dir, "diff_%s.txt" % metric), diff_mat, fmt='%.3f')

				embedding = tsne.fit_transform(sym_dist_mat)
				out_im = os.path.join(out_dir, "embedding_%s.png" % metric)
				graph_embedding(out_im, embedding, colors, sizes)

				nebs = num_knns(sym_dist_mat, groups)
				out_im = os.path.join(out_dir, "nebs_all_%s.png" % metric)
				graph_nebs(out_im, nebs, rnebs)

				# knns by group
				all_colors = set(colors)
				sub_out_dir = os.path.join(out_dir, "group_nebs_%s" % metric)
				safe_mkdir(sub_out_dir)
				print "Knns by group"
				for color in all_colors:
					sub_group = np.asarray([ [c1 == c2 and c1 == color for c2 in colors] for c1 in colors])

					nebs = num_knns(sym_dist_mat, sub_group)
					rnebs = get_random_knn(sub_group)
					out_im = os.path.join(sub_out_dir, "%s_%s.png" % (color_names[color], metric))
					graph_nebs(out_im, nebs, rnebs)

				# pairwise embeddings 
				sub_out_dir = os.path.join(out_dir, "pairwise_%s" % metric)
				safe_mkdir(sub_out_dir)
				print "Pairwise"
				for idx1, color1 in enumerate(all_colors):
					print "Pairwise", idx1
					for idx2, color2 in enumerate(all_colors):
						if idx1 >= idx2:
							continue
						sub_groups = np.asarray([ [c1 == c2 and (c1 == color1 or c1 == color2) for c2 in colors] for c1 in colors])
						indices = []
						sub_colors = []
						sub_sizes = []
						idx = 0
						for c,s in zip(colors, sizes):
							if c == color1 or c == color2:
								indices.append(idx)
								sub_colors.append(c)
								sub_sizes.append(s)
							idx += 1
						sub_sym_dist_mat = sym_dist_mat[indices, :]
						sub_sym_dist_mat = sub_sym_dist_mat[:, indices]

						embedding = tsne.fit_transform(sub_sym_dist_mat)
						out_im = os.path.join(sub_out_dir, "embedding_%s_%s.png" % (color_names[color1], color_names[color2]))
						graph_embedding(out_im, embedding, sub_colors, sub_sizes)


