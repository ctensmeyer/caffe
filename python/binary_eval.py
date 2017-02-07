
import os
import sys
import numpy as np
import cv2


def h_mean(x1, x2):
	return 2 * x1 * x2 / (x1 + x2) if (x1 + x2) else 0

def invert(im):
	return np.max(im) - im

def measure_acc(im, gt):
	num_errors = np.count_nonzero(im - gt)	
	total = im.shape[0] * im.shape[1] 
	num_correct = total - num_errors
	return 100 * float(num_correct) / total


def measure_psnr(im, gt):
	diff = im - gt
	mse = np.mean(diff * diff)
	_max = np.max(gt)
	return 10 * np.log10(_max * _max / mse)


def measure_drd(im, gt):
	return 0.


# convention that 0 is foreground and != 0 is background
def measure_fmeasure(im, gt):
	tp = np.logical_and(im == 0, gt == 0).sum()
	tn = np.logical_and(im != 0, gt != 0).sum()

	fp = np.logical_and(im == 0, gt != 0).sum()
	fn = np.logical_and(im != 0, gt == 0).sum()

	assert tp + tn + fp + fn == im.shape[0] * im.shape[1]
	recall = 100 * float(tp) / (tp + fn) if (tp + fn) else 0
	precision = 100 * float(tp) / (tp + fp) if (tp + fp) else 0
	f_measure = h_mean(recall, precision)
	return f_measure, precision, recall


def measure_psuedo_fmeasure(im, gt, r_weights, p_weights):
	# convention that >0 is foreground and 0 is background
	# therefore invert the input images
	im = invert(im)
	gt = invert(gt)

	denum = float((gt * r_weights).sum()) 
	recall = (100. / 255) * (im * gt * r_weights).sum() / denum if denum else 0

	denum = float((im * p_weights).sum()) 
	precision = (100. / 255) * (im * gt * p_weights).sum() / denum if denum else 0
	f_measure = h_mean(recall, precision)
	return f_measure, precision, recall
	

def get_metrics(predict_fn, gt_fn, recall_fn, precision_fn):
	predict_im = cv2.imread(predict_fn, -1).astype(np.int32)
	gt_im = cv2.imread(gt_fn, -1).astype(np.int32)
	recall_weights = np.loadtxt(recall_fn).reshape(gt_im.shape)
	precision_weights = np.loadtxt(precision_fn).reshape(gt_im.shape) + 1

	assert predict_im.shape == gt_im.shape

	accuracy = measure_acc(predict_im, gt_im)
	psnr = measure_psnr(predict_im, gt_im)
	drd = measure_drd(predict_im, gt_im)
	f, p, r = measure_fmeasure(predict_im, gt_im)
	pf, pp, pr = measure_psuedo_fmeasure(predict_im, gt_im, recall_weights, precision_weights)

	return pf, pp, pr, f, p, r, psnr, accuracy

def main(predict_dir, pr_dat_dir, out_file, summary_file):
	fd = open(out_file, 'w')
	all_metrics = list()
	for fn in os.listdir(predict_dir):
		print fn
		predict_fn = os.path.join(predict_dir, fn)
		gt_fn = os.path.join(pr_dat_dir, fn)
		base = os.path.splitext(fn)[0]
		recall_fn = os.path.join(pr_dat_dir, base + "_RWeights.dat")
		precision_fn = os.path.join(pr_dat_dir, base + "_PWeights.dat")
		metrics = get_metrics(predict_fn, gt_fn, recall_fn, precision_fn)
		all_metrics.append(metrics)
		fd.write("%s  %s\n" % (fn, "  ".join(map(lambda f: "%.4f" % f, metrics))))

	fd.close()

	# summary stuff
	fd = open(summary_file, 'w')
	avg_metrics = np.mean(all_metrics, axis=0)
	fd.write("avg:  %s\n" % "  ".join(map(lambda f: "%.4f" % f, avg_metrics)))
	fd.close()
		

if __name__ == "__main__":
	predict_dir = sys.argv[1]
	pr_dat_dir = sys.argv[2]
	out_file = sys.argv[3]
	summary_file = sys.argv[4]
	main(predict_dir, pr_dat_dir, out_file, summary_file)
	
