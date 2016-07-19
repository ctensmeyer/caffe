// Copyright 2013 Yangqing Jia

#include <iostream>  // NOLINT(readability/streams)
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

int next_power_of_2(int a) {
  int pow = 1;
  while (pow < a) {
    pow *= 2;
  }
  return pow;
}

template <typename Dtype>
void PaddingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->Reshape(bottom, top);
}

template <typename Dtype>
void PaddingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  pad_h_top_ = pad_h_bottom_ = pad_w_left_ = pad_w_right_ = this->layer_param_.padding_param().pad(); 
  if (pad_h_top_ == 0) {
    pad_h_top_ = pad_h_bottom_ = this->layer_param_.padding_param().pad_h();
    pad_w_left_ = pad_w_right_ = this->layer_param_.padding_param().pad_w();
  }
  if (pad_h_top_ == 0) {
    pad_h_top_ = this->layer_param_.padding_param().pad_h_top();
    pad_h_bottom_ = this->layer_param_.padding_param().pad_h_bottom();
  }
  if (pad_w_left_ == 0) {
    pad_w_left_ = this->layer_param_.padding_param().pad_w_left();
    pad_w_right_ = this->layer_param_.padding_param().pad_w_right();
  }
  num_ = bottom[0]->num();
  channel_ = bottom[0]->channels();
  height_in_ = bottom[0]->height();
  width_in_ = bottom[0]->width();

  if (this->layer_param_.padding_param().pad_to_power_of_2()) {
    // height
	height_out_ = next_power_of_2(height_in_);
	int total_pad = height_out_ - height_in_;
	pad_h_top_ = pad_h_bottom_ = total_pad / 2;
	if (total_pad % 2) {
	  pad_h_bottom_++;
	}

    // width
	width_out_ = next_power_of_2(width_in_);
	total_pad = width_out_ - width_in_;
	pad_w_left_ = pad_w_right_ = total_pad / 2;
	if (total_pad % 2) {
	  pad_w_right_++;
	}
  }

  CHECK_EQ(bottom.size(), 1) << "Padding Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Padding Layer takes a single blob as output.";
  num_ = bottom[0]->num();
  channel_ = bottom[0]->channels();
  height_in_ = bottom[0]->height();
  width_in_ = bottom[0]->width();
  height_out_ = height_in_ + pad_h_top_ + pad_h_bottom_;
  width_out_ = width_in_ + pad_w_left_ + pad_w_right_;
  top[0]->Reshape(num_, channel_, height_out_, width_out_);
}

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  memset(top_data, 0, sizeof(Dtype) * top[0]->count());
  // In short, top[n, c, h, w] = bottom[n, c, h-pad, w-pad] if in range
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channel_; ++c) {
      for (int h = 0; h < height_in_; ++h) {
        // copy the width part
        memcpy(
            top_data + ((n * channel_ + c) * height_out_ + h + pad_h_top_)
                * width_out_ + pad_w_left_,
            bottom_data + ((n * channel_ + c) * height_in_ + h) * width_in_,
            sizeof(Dtype) * width_in_);
      }
    }
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channel_; ++c) {
      for (int h = 0; h < height_in_; ++h) {
        // copy the width part
        memcpy(
            bottom_diff + ((n * channel_ + c) * height_in_ + h) * width_in_,
            top_diff + ((n * channel_ + c) * height_out_ + h + pad_h_top_)
                * width_out_ + pad_w_left_,
            sizeof(Dtype) * width_in_);
      }
    }
  }
}
#ifdef CPU_ONLY
STUB_GPU(PaddingLayer);
#endif

INSTANTIATE_CLASS(PaddingLayer);
REGISTER_LAYER_CLASS(Padding);

}  // namespace caffe
