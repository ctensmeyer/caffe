import sys
import cv2
import numpy as np
#import matplotlib.pyplot as plt

def extract_seam(img_en, diag_cost=1.414):

    img_en = img_en.copy()

    joined = np.zeros((3, img_en.shape[1]), dtype=np.float64)
    second_index = np.arange(joined.shape[1])

    cum_en = np.zeros_like(img_en)
    n = np.inf
    cum_en = np.pad(cum_en, ((0,0),(1,1)), mode='constant', constant_values=((n,n),(n,n)))

    prev_slice = cum_en[0]
    back = np.zeros(img_en.shape, dtype=np.uint8)

    for i in range(0, img_en.shape[0]):
        joined[0] = prev_slice[:-2] + img_en[i] * diag_cost
        joined[1] = prev_slice[1:-1] + img_en[i]
        joined[2] = prev_slice[2:] + img_en[i] * diag_cost

        joined_idx = np.argmin(joined, axis=0)
        back[i] = joined_idx

        cum_en[i][1:-1] = joined[(joined_idx, second_index)]
        prev_slice = cum_en[i]

    seam = np.zeros(img_en.shape[0], dtype=np.uint64)
    col = cum_en[-1,1:-1].argmin()
    for j in range(img_en.shape[0]-1, -1, -1):
        assert col >= 0
        seam[j] = col
        col += back[j,col] - 1
        col = max(col, 0)
        col = min(col, back.shape[1]-1)
    return seam

def paint_seam(img_shape, seam):

    img_new = np.full(img_shape, 255, dtype=np.uint8)
    for i in range(img_shape[0]):
        img_new[i,seam[i]] = 0

    return img_new

def paint_seam_float(img_shape, seam):

    img_new = np.full(img_shape, 0, dtype=np.float32)
    for i in range(img_shape[0]):
        img_new[i,seam[i]] = 1.0

    return img_new

#if __name__ == "__main__":
#
#    img = np.zeros((256,256), np.uint8)
#
#
#    baseline = [[50,150],[100,160],[200,145],[350,155]]
#    pts = np.array(baseline, np.int32)
#    pts = pts.reshape((-1,1,2))
#    cv2.polylines(img,[pts],False,255)
#    #
#    dist = cv2.distanceTransform(255 - img,cv2.DIST_L2,5)
#    dist[dist > 20] = 20
#    dist[dist != 20] = 0
#
#    dist = (dist - dist.min())/(dist.max() -  dist.min())
#
#    dist = cv2.distanceTransform((1.0 - dist).astype(np.uint8),cv2.DIST_L2,5)
#    dist = (dist - dist.min())/(dist.max() -  dist.min())
#    dist = 1.0 - dist
#
#    #
#    # plt.imshow(dist)
#    # plt.show()
#
#    # dist = img
#    # dist = (255 - dist).astype(float) / 255
#    dist[:,150:200] = 1.0
#
#    noise = np.random.normal(0.01, 0.05, size=dist.shape)
#    dist = dist + noise
#
#    dist = dist.T
#
#    import time
#    start = time.time()
#    iters = 20
#    for i in range(iters):
#        # seams = extractSeam(dist, False)
#        seams = extract_seam(dist)
#    print iters/(time.time() - start)
#    # raw_input()
#
#
#    new_img = paint_seam(dist, seams)
#
#    # print dist.max()
#    # print new_img.max()
#    #
#    # print dist.sum()
#    # print new_img.sum()
#
#    plt.imshow((1.0 - dist) + new_img)
#    plt.show()
