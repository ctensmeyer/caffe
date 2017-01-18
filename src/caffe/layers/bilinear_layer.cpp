// Copyright 2013 Yangqing Jia

#include <iostream>  // NOLINT(readability/streams)
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BilinearInterpolationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //this->Reshape(bottom, top);
  //CHECK_EQ(bottom[1]->count(), 2) << "BilinearInterpolationLayer requires the output height/width";
}

template <typename Dtype>
void BilinearInterpolationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //CHECK_EQ(bottom[1]->count(), 2) << "BilinearInterpolationLayer requires the output height/width";

  num_ = bottom[0]->num();
  channel_ = bottom[0]->channels();
  height_in_ = bottom[0]->height();
  width_in_ = bottom[0]->width();
  height_out_ = (int) bottom[1]->shape()[2];
  width_out_ = (int) bottom[1]->shape()[3];
  /*
  DLOG(ERROR) << num_ << " " << channel_ << " " << height_in_ << " " <<
  	width_in_ << " " <<  height_out_ << " " << width_out_;
  */
  CHECK_GT(height_out_, 0) << "Specified height must be positive integer";
  CHECK_GT(width_out_, 0) << "Specified width must be positive integer";
  top[0]->Reshape(num_, channel_, height_out_, width_out_);
}

template <typename Dtype>
void BilinearInterpolationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  Dtype height_factor = (height_in_ - 1) / (Dtype) (height_out_ - 1);
  Dtype width_factor = (width_in_ - 1) / (Dtype) (width_out_ - 1);
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channel_; ++c) {
      for (int h = 0; h < height_out_; ++h) {
	      Dtype h_orig = h * height_factor;
		  int h_floor = (int) h_orig;
		  int h_ceil = (int) (h_orig + 1);
		  int bottom_index = ((n * channel_ + c) * height_in_ + h_floor);
		  int top_index = ((n * channel_ + c) * height_out_ + h);
		  Dtype h_ceil_orig = h_ceil - h_orig;
		  Dtype h_orig_floor = h_orig - h_floor;
	    for (int w = 0; w < width_out_; ++w) {
		  Dtype w_orig = w * width_factor;
		  int w_floor = (int) w_orig;
		  int w_ceil = (int) (w_orig + 1);

		  Dtype f00 = bottom_data[bottom_index * width_in_ + w_floor];
		  Dtype f01 = bottom_data[bottom_index * width_in_ + w_ceil];
		  Dtype f10 = bottom_data[(bottom_index + 1) * width_in_ + w_floor];
		  Dtype f11 = bottom_data[(bottom_index + 1) * width_in_ + w_ceil];
		  Dtype a00 = h_ceil_orig  * (w_ceil - w_orig);
		  Dtype a01 = h_ceil_orig  * (w_orig - w_floor);
		  Dtype a10 = h_orig_floor * (w_ceil - w_orig);
		  Dtype a11 = h_orig_floor * (w_orig - w_floor);
		  Dtype interp_val = a00 * f00 +
		  					 a10 * f10 +
							 a01 * f01 +
							 a11 * f11;
		  top_data[top_index * width_out_ + w] = interp_val;

		  /*
		  DLOG(ERROR) << "h/w/h_orig/w_orig " << h << " " << w << " " << h_orig << " " << w_orig;
		  DLOG(ERROR) << "\th_floor/h_ceil/w_floor/w_ceil " << h_floor << " " << h_ceil << " " << w_floor << " " << w_ceil;
		  DLOG(ERROR) << "\tf00/f10/f01/f11 " << f00 << " " << f10 << " " << f01 << " " << f11;
		  DLOG(ERROR) << "\tinterp " << interp_val;
		  */

		}
      }
    }
  }
  /*
  DLOG(ERROR) << "\tbottom_data: ";
  DLOG(ERROR) << bottom_data[0] << " " << bottom_data[1];
  DLOG(ERROR) << bottom_data[2] << " " << bottom_data[3];

  DLOG(ERROR) << "\ttop_data: ";
  DLOG(ERROR) << top_data[0] << " " << top_data[1] << " " << top_data[2] << " " << top_data[3];
  DLOG(ERROR) << top_data[4] << " " << top_data[5] << " " << top_data[6] << " " << top_data[7];
  DLOG(ERROR) << top_data[8] << " " << top_data[9] << " " << top_data[10] << " " << top_data[11];
  */

/*
  DLOG(ERROR) << "\tbottom_data: ";
  DLOG(ERROR) << bottom_data[0] << " " << bottom_data[1] << " " << bottom_data[2];
  DLOG(ERROR) << bottom_data[3] << " " << bottom_data[4] << " " << bottom_data[5];
  DLOG(ERROR) << bottom_data[6] << " " << bottom_data[7] << " " << bottom_data[8];

  DLOG(ERROR) << "\ttop_data: ";
  DLOG(ERROR) << top_data[0] << " " << top_data[1] << " " << top_data[2] << " " << top_data[3] << " " << top_data[4];
  DLOG(ERROR) << top_data[5] << " " << top_data[6] << " " << top_data[7] << " " << top_data[8] << " " << top_data[9];
  DLOG(ERROR) << top_data[10] << " " << top_data[11] << " " << top_data[12] << " " << top_data[13] << " " << top_data[14];
  DLOG(ERROR) << top_data[15] << " " << top_data[16] << " " << top_data[17] << " " << top_data[18] << " " << top_data[19];
*/

}

template <typename Dtype>
void BilinearInterpolationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  memset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count());

  Dtype height_factor = (height_in_ - 1) / (Dtype) (height_out_ - 1);
  Dtype width_factor = (width_in_ - 1) / (Dtype) (width_out_ - 1);
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channel_; ++c) {
      for (int h = 0; h < height_out_; ++h) {
	      Dtype h_orig = h * height_factor;
		  int h_floor = (int) h_orig;
		  int h_ceil = (int) (h_orig + 1);
		  int bottom_index = ((n * channel_ + c) * height_in_ + h_floor);
		  int top_index = ((n * channel_ + c) * height_out_ + h);
		  Dtype h_ceil_orig = h_ceil - h_orig;
		  Dtype h_orig_floor = h_orig - h_floor;
	    for (int w = 0; w < width_out_; ++w) {
		  Dtype w_orig = w * width_factor;
		  int w_floor = (int) w_orig;
		  int w_ceil = (int) (w_orig + 1);

		  Dtype d_out = top_diff[top_index * width_out_ + w];
		  Dtype a00 = h_ceil_orig  * (w_ceil - w_orig);
		  Dtype a01 = h_ceil_orig  * (w_orig - w_floor);
		  Dtype a10 = h_orig_floor * (w_ceil - w_orig);
		  Dtype a11 = h_orig_floor * (w_orig - w_floor);

		  bottom_diff[bottom_index * width_in_ + w_floor] += a00 * d_out;
		  bottom_diff[bottom_index * width_in_ + w_ceil] += a01 * d_out;
		  bottom_diff[(bottom_index + 1) * width_in_ + w_floor] += a10 * d_out;
		  bottom_diff[(bottom_index + 1) * width_in_ + w_ceil] += a11 * d_out;

		}
      }
    }
  }
  /*
  DLOG(ERROR) << "\ttop_diffs: ";
  DLOG(ERROR) << top_diff[0] << " " << top_diff[1] << " " << top_diff[2] << " " << top_diff[3];
  DLOG(ERROR) << top_diff[4] << " " << top_diff[5] << " " << top_diff[6] << " " << top_diff[7];
  DLOG(ERROR) << top_diff[8] << " " << top_diff[9] << " " << top_diff[10] << " " << top_diff[11];

  DLOG(ERROR) << "\tbottom_diffs: ";
  DLOG(ERROR) << bottom_diff[0] << " " << bottom_diff[1];
  DLOG(ERROR) << bottom_diff[2] << " " << bottom_diff[3];
  */

/*
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
#ifdef CPU_ONLY
STUB_GPU(BilinearInterpolationLayer);
#endif

INSTANTIATE_CLASS(BilinearInterpolationLayer);
REGISTER_LAYER_CLASS(BilinearInterpolation);

}  // namespace caffe
