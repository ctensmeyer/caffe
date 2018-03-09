#include <algorithm>
#include <cfloat>
#include <vector>
#include <math.h>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/shift_layer.hpp"

namespace caffe {

template <typename Dtype>
void ShiftLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  h_shift_ = this->layer_param().shift_param().h_shift();
  w_shift_ = this->layer_param().shift_param().w_shift();
}

template <typename Dtype>
void ShiftLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape());
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
}

template <typename Dtype>
void ShiftLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const int count = top[0]->count();
  caffe_set(count, Dtype(0), top_data);

  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < height_; ++h) {
	    int from_h = h - h_shift_;
		if (from_h < 0 || from_h >= height_) {
		  continue;
		}
        for (int w = 0; w < width_; ++w) {
	      int from_w = w - w_shift_;
		  if (from_w >= 0 && from_w < width_) {
		    int from_idx = ((n * channels_ + c) * height_ + from_h) * width_ + from_w;
			int to_idx = ((n * channels_ + c) * height_ + h) * width_ + w;
			top_data[to_idx] = bottom_data[from_idx];
		  }
	    }
      }
    }
  }
}

template <typename Dtype>
void ShiftLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    const int bottom_count = bottom[0]->count();
    caffe_set(bottom_count, Dtype(0), bottom_diff);

    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h) {
          int to_h = h - h_shift_;
          if (to_h < 0 || to_h >= height_) {
      	    continue;
      	  }
          for (int w = 0; w < width_; ++w) {
            int to_w = w - w_shift_;
            if (to_w >= 0 && to_w < width_) {
      	      int from_idx = ((n * channels_ + c) * height_ + h) * width_ + w;
      		  int to_idx = ((n * channels_ + c) * height_ + to_h) * width_ + to_w;
      		  bottom_diff[to_idx] = top_diff[from_idx];
      	    }
          }
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ShiftLayer);
#endif

INSTANTIATE_CLASS(ShiftLayer);
REGISTER_LAYER_CLASS(Shift);

}  // namespace caffe

