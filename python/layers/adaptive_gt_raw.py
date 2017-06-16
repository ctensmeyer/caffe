#import cv2
import numpy as np
#import matplotlib.pyplot as plt
import sys
from seam_carve import extract_seam, paint_seam_float

def adaptive_gt(pred, gt, tolerance=20, alpha=0.001, beta=1.414):

    gt = gt.astype(np.uint8)

    output = np.full(pred.shape[:2], 0.0, np.float32)

    all_labels, all_dist, all_ep_dist = gt[0,:,:], gt[1,:,:], gt[2,:,:]

    clipped_label = all_labels.copy()
    clipped_label[all_dist > tolerance] = 0

    clipped_ep = all_labels.copy()
    clipped_ep[all_ep_dist > tolerance] = 0

    for cnt, i in enumerate(np.unique(clipped_label)):
        #Skip background
        if i == 0:
            continue

        idxs = np.where(clipped_label==i)

        max0 = idxs[0].max()
        max1 = idxs[1].max()

        min0 = idxs[0].min()
        min1 = idxs[1].min()

        gt_region = clipped_label[min0:max0+1, min1:max1+1].copy()

        cost_region = all_dist[min0:max0+1, min1:max1+1].copy()
        cost_region = cost_region.astype(np.float32)

        pred_region = pred[min0:max0+1, min1:max1+1].copy()
        pred_region[gt_region != i] = -np.inf

        # noise = np.abs(np.random.normal(0.0, 0.0001, size=pred_region.shape))
        # pred_region = pred_region + noise

        energy = (1.0 - pred_region) + alpha * cost_region

        seams = extract_seam(energy.T, diag_cost=beta)
        seam_img = paint_seam_float(pred_region.T.shape, seams).T

        #Handle end points
        ep_region = clipped_ep[min0:max0+1, min1:max1+1].copy()
        ep_seam = pred_region.copy()
        ep_seam[ep_region != i] = 0
        ep_seam = ep_seam * seam_img

        #Handle everywhere that is not an endpoint
        no_ep_seam = seam_img.copy()
        no_ep_seam[ep_region == i] = 0

        #Join end points and non end points
        final_seam = no_ep_seam + ep_seam

        output[min0:max0+1, min1:max1+1] = np.maximum(output[min0:max0+1, min1:max1+1], final_seam)


    return output

#if __name__ == "__main__":
#    pred_path = sys.argv[1]
#    gt_path = sys.argv[2]
#    output_path = sys.argv[3]
#
#    pred = cv2.imread(pred_path, 0)
#    gt = cv2.imread(gt_path).astype(np.float32)
#
#    # x = 1500
#    # y = 1500
#    # w = 256
#    # pred = pred[x:x+w,y:y+w]
#    # gt = gt[x:x+w,y:y+w]
#
#    pred = cv2.blur(pred,(1,7))
#    pred = pred.astype(np.float32) / 255
#
#    new_gt = adaptive_gt(pred, gt)
#    kernel = np.ones((7,7),np.uint8)
#    new_gt = cv2.dilate(new_gt,kernel,iterations = 1)
#    # plt.imshow(new_gt)
#    # plt.show()
#
#    dilate_factor = 7
#    colored = np.zeros(gt.shape, dtype=np.uint8)
#
#    cost_region = gt[:,:,1]
#    cost_region[cost_region>dilate_factor/2] = 255
#    cost_region[cost_region<=dilate_factor/2] = 0
#
#
#
#    colored[:,:,0] = cost_region
#    colored[:,:,1] = 255 - pred * 255
#    colored[:,:,2] = 255 - new_gt * 255
#
#    cv2.imwrite(output_path, colored)
