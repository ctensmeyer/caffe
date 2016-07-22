// Copyright 2013 Yangqing Jia

#include <iostream>  // NOLINT(readability/streams)
#include <vector>
#include <stdio.h>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BilinearForward(const int count, const Dtype* bottom_data, Dtype* top_data,
					const int num, const int channels, const int in_height, const int in_width, 
					const int out_height, const int out_width) {
  Dtype height_factor = (in_height - 1) / (Dtype) (out_height - 1);
  Dtype width_factor = (in_width - 1) / (Dtype) (out_width - 1);
  int num_size = channels * out_height * out_width;
  int channel_size = out_height * out_width;
  int height_size = out_width;
  CUDA_KERNEL_LOOP(index, count) {
    // recover n, c, h, w from index
	int n = index / num_size;
	int c = (index - (n * num_size)) / channel_size;
	int h = (index - (n * num_size) - (c * channel_size)) / height_size;
	int w = (index - (n * num_size) - (c * channel_size) - (h * height_size));
	//printf("index: %d n: %d c: %d h: %d w: %d\n", index, n, c, h, w);

	// height computations
    // TODO: precompute offsets in 1D tables
	Dtype h_orig = h * height_factor;
	int h_floor = (int) h_orig;
	int h_ceil = (int) (h_orig + 1);
	int bottom_index = ((n * channels + c) * in_height + h_floor);
	int top_index = ((n * channels + c) * out_height + h);
	Dtype h_ceil_orig = h_ceil - h_orig;
	Dtype h_orig_floor = h_orig - h_floor;

	// width computations
	Dtype w_orig = w * width_factor;
	int w_floor = (int) w_orig;
	int w_ceil = (int) (w_orig + 1);

	Dtype f00 = bottom_data[bottom_index * in_width + w_floor];
	Dtype f01 = bottom_data[bottom_index * in_width + w_ceil];
	Dtype f10 = bottom_data[(bottom_index + 1) * in_width + w_floor];
	Dtype f11 = bottom_data[(bottom_index + 1) * in_width + w_ceil];
	Dtype a00 = h_ceil_orig  * (w_ceil - w_orig);
	Dtype a01 = h_ceil_orig  * (w_orig - w_floor);
	Dtype a10 = h_orig_floor * (w_ceil - w_orig);
	Dtype a11 = h_orig_floor * (w_orig - w_floor);
	Dtype interp_val = a00 * f00 +
	  				   a10 * f10 +
	  				   a01 * f01 +
	  				   a11 * f11;
	top_data[top_index * out_width + w] = interp_val;
  }
}


template <typename Dtype>
void BilinearInterpolationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  int count = top[0]->count();

  BilinearForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, num_, channel_, height_in_, width_in_,
	  height_out_, width_out_);
  CUDA_POST_KERNEL_CHECK;

/*
  top_data = top[0]->mutable_cpu_data();
  bottom_data = bottom[0]->cpu_data();
  DLOG(ERROR) << "\tbottom_data: ";
  DLOG(ERROR) << bottom_data[0] << " " << bottom_data[1];
  DLOG(ERROR) << bottom_data[2] << " " << bottom_data[3];

  DLOG(ERROR) << "\ttop_data: ";
  DLOG(ERROR) << top_data[0] << " " << top_data[1] << " " << top_data[2] << " " << top_data[3];
  DLOG(ERROR) << top_data[4] << " " << top_data[5] << " " << top_data[6] << " " << top_data[7];
  DLOG(ERROR) << top_data[8] << " " << top_data[9] << " " << top_data[10] << " " << top_data[11];
*/

}
template <typename Dtype>
__global__ void BilinearBackward(const int count, const Dtype* top_diff, Dtype* bottom_diff,
					const int num, const int channels, const int in_height, const int in_width, 
					const int out_height, const int out_width) {
  Dtype height_factor = (out_height - 1) / (Dtype) (in_height - 1);
  Dtype width_factor = (out_width - 1) / (Dtype) (in_width - 1);
  Dtype height_factor_2 = (in_height - 1) / (Dtype) (out_height - 1);
  Dtype width_factor_2 = (in_width - 1) / (Dtype) (out_width - 1);
  int num_size = channels * in_height * in_width;
  int channel_size = in_height * in_width;
  int height_size = in_width;
  CUDA_KERNEL_LOOP(index, count) {
    // recover n, c, h, w for input blob from index
	int n = index / num_size;
	int c = (index - (n * num_size)) / channel_size;
	int h = (index - (n * num_size) - (c * channel_size)) / height_size;
	int w = (index - (n * num_size) - (c * channel_size) - (h * height_size));
	//printf("index: %d n: %d c: %d h: %d w: %d\n", index, n, c, h, w);

    // forward mapping of coords

	// find the minimum h coordinate in the output image that the input h reaches
	int h_out_min = ((int) (max((h - 1) *  height_factor, -1.) + 1));
	//int h_out_max = (int) min(h_out + height_factor, out_height - 1.);
	int h_out_max = (int) min((h + 1) * height_factor, out_height - 1.);

	int w_out_min = ((int) (max((w - 1) * width_factor, -1.) + 1));
	int w_out_max = (int) min((w + 1) *width_factor, out_width - 1.);

	Dtype d_in = 0.;
	/*
	printf("(%d, %d): h_out: [%d, %d], w_out: [%d, %d]\n", h, w, h_out_min, h_out_max, w_out_min, w_out_max);
	printf("\t%f\n", (h-1) * height_factor);
	*/
	for (int h2 = h_out_min; h2 <= h_out_max; h2++) {
	  Dtype h_orig = h2 * height_factor_2;
	  Dtype h_mult = 0;
	  if (h_orig < h) {
	    h_mult = 1 - h + h_orig;
	  } else {
	    h_mult = 1 - h_orig + h;
	  }
	  for (int w2 = w_out_min; w2 <= w_out_max; w2++) {
	    Dtype w_orig = w2 * width_factor_2;
	    Dtype w_mult = 0;
	    if (w_orig < w) {
	      w_mult = 1 - w + w_orig;
	    } else {
	      w_mult = 1 - w_orig + w;
	    }
		Dtype coeff = h_mult * w_mult;
	    int top_index = ((n * channels + c) * out_height + h2) * out_width + w2;
		Dtype err = top_diff[top_index];
		d_in += coeff * err;
	  }
	}
	int bottom_index = ((n * channels + c) * in_height + h) * in_width + w;
	bottom_diff[bottom_index] = d_in;
  }
}

template <typename Dtype>
void BilinearInterpolationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    int count = bottom[0]->count();

    BilinearBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_diff, num_, channel_, height_in_, width_in_,
        height_out_, width_out_);
    CUDA_POST_KERNEL_CHECK;

/*
    top_diff = top[0]->cpu_diff();
	bottom_diff = bottom[0]->mutable_cpu_diff();
    DLOG(ERROR) << "\ttop_diffs: ";
    DLOG(ERROR) << top_diff[0] << " " << top_diff[1] << " " << top_diff[2] << " " << top_diff[3];
    DLOG(ERROR) << top_diff[4] << " " << top_diff[5] << " " << top_diff[6] << " " << top_diff[7];
    DLOG(ERROR) << top_diff[8] << " " << top_diff[9] << " " << top_diff[10] << " " << top_diff[11];

    DLOG(ERROR) << "\tbottom_diffs: ";
    DLOG(ERROR) << bottom_diff[0] << " " << bottom_diff[1];
    DLOG(ERROR) << bottom_diff[2] << " " << bottom_diff[3];
*/
/*
  top_diff = top[0]->cpu_diff();
  bottom_diff = bottom[0]->mutable_cpu_diff();
  DLOG(ERROR) << "\ttop_diff: ";
  DLOG(ERROR) << top_diff[0] << " " << top_diff[1] << " " << top_diff[2] << " " << top_diff[3] << " " << top_diff[4];
  DLOG(ERROR) << top_diff[5] << " " << top_diff[6] << " " << top_diff[7] << " " << top_diff[8] << " " << top_diff[9];
  DLOG(ERROR) << top_diff[10] << " " << top_diff[11] << " " << top_diff[12] << " " << top_diff[13] << " " << top_diff[14];
  DLOG(ERROR) << top_diff[15] << " " << top_diff[16] << " " << top_diff[17] << " " << top_diff[18] << " " << top_diff[19];

  DLOG(ERROR) << "\tbottom_diff: ";
  DLOG(ERROR) << bottom_diff[0] << " " << bottom_diff[1] << " " << bottom_diff[2];
  DLOG(ERROR) << bottom_diff[3] << " " << bottom_diff[4] << " " << bottom_diff[5];
  DLOG(ERROR) << bottom_diff[6] << " " << bottom_diff[7] << " " << bottom_diff[8];
*/

  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BilinearInterpolationLayer);
}  // namespace caffe
