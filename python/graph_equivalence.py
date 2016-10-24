
import os
import sys
import ast
from utils import safe_mkdir
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import sklearn.manifold.TSNE

METRICS = ['accuracy', 'agreement', 'avg_jsd', 'avg_l2', 'avg_scaled_l2']
MODEL_TYPES = ['linear']
LOSS_TYPES = ['l2']
SPLITS = ['train', 'test']

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
'v_mirror' : ('brown' 50),

'perspective_1' : ('cyan', 20),
'perspective_2' : ('cyan', 35),
'perspective_3' : ('cyan', 50),
'perspective_4' : ('cyan' 65),

'rotation_5' : ('olive', 20),
'rotation_10' : ('olive', 35),
'rotation_15' : ('olive', 50),
'rotation_20' : ('olive', 65),

'shear_5' : ('orange', 20),
'shear_10' : ('orange', 35),
'shear_15' : ('orange', 50),
'shear_20' : ('orange', 65),

}


def graph_embedding(out_file, Y, colors, sizes):
	plt.scatter(Y[:,0], Y[:,1], c=colors, s=sizes)
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
				

ROOT='/fslhome/waldol1/fsl_groups/fslg_nnml/compute/experiments/preprocessing/nets'
group_file = sys.argv[1]
out_dir = sys.argv[2]

net_dirs = open(group_file, 'r').readlines()
net_dirs = map(lambda s: s.rstrip(), net_dirs)
num_net_dirs = len(net_dirs)

colors, sizes = get_color_sizes(net_dirs)

tsne = sklearn.manifold.TSNE(n_components=2, method='exact', metric='precomputed')

for split in SPLITS:
	for loss in LOSS_TYPES:
		for model_type in MODEL_TYPES:
			#out_dir = os.path.join(out_dir, split, loss, model_type)
			out_dir = os.path.join(out_dir, split)
			safe_mkdir(out_dir)
			for metric in METRICS:

				dist_mat = np.zeros( (num_net_dirs, num_net_dirs), dtype=float)
				for idx1, net_dir1 in enumerate(net_dirs):
					result_dir = os.path.join(ROOT, net_dir1, 'equivalence/results')
					for idx2, net_dir2 in enumerate(net_dirs):
						fn = net_dir2.replace('/', '_') + '.txt'
						result_file = os.path.join(result_dir, fn)
						results = ast.literal_eval(open(result_file, 'r').read())
						dist_mat[idx1,idx2] = results[split][model_type][loss][metric]

				np.savetxt(os.path.join(out_dir, "raw_dist_%s.txt" % metric), dist_mat, fmt='%.3f')
				sym_dist_mat = (dist_mat + dist_mat.T) / 2
				np.savetxt(os.path.join(out_dir, "sym_dist_%s.txt" % metric), sym_dist_mat, fmt='%.3f')
				diff_mat = dist_mat - dist_mat.T
				np.savetxt(os.path.join(out_dir, "diff_%s.txt" % metric), diff_mat, fmt='%.4f')

				embedding = tsne.fit_transform(sym_dist_mat)
				out_im = os.path.join(out_dir, "embedding_%s.png" % metric)
				graph_embedding(out_im, embedding, colors, sizes)



					

