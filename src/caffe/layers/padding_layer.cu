// Copyright 2013 Yangqing Jia

#include <iostream>  // NOLINT(readability/streams)
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PaddingForward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int pad_h_top, const int pad_w_left, const int height_out, const int width_out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) {
    int w = index % width_in;
    index /= width_in;
    int h = index % height_in;
    index /= height_in;
    int c = index % channel;
    index /= channel;
    out[((index * channel + c) * height_out + h + pad_h_top) * width_out + pad_w_left + w] =
        in[((index * channel + c) * height_in + h) * width_in + w];
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // First, set all data to be zero for the boundary pixels
  CUDA_CHECK(cudaMemset(top_data, 0, sizeof(Dtype) * top[0]->count()));
  // NOLINT_NEXT_LINE(whitespace/operators)
  PaddingForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, num_, channel_, height_in_, width_in_,
      pad_h_top_, pad_w_left_, height_out_, width_out_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void PaddingBackward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int pad_h_top, const int pad_w_left, const int height_out, const int width_out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) {
    int w = index % width_in;
    index /= width_in;
    int h = index % height_in;
    index /= height_in;
    int c = index % channel;
    index /= channel;
    out[((index * channel + c) * height_in + h) * width_in + w] =
      in[((index * channel + c) * height_out + h + pad_h_top) *
         width_out + pad_w_left + w];
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PaddingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_diff, num_, channel_, height_in_, width_in_,
        pad_h_top_, pad_w_left_, height_out_, width_out_);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PaddingLayer);

}  // namespace caffe
