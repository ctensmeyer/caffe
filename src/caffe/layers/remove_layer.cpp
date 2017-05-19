#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void RemoveLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RemoveLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) <<
      "Remove layer inputs must have the same num.";
  CHECK_EQ(bottom[1]->count(), bottom[1]->num()) <<
      "Remove layer index blob must be count == num";
  CHECK_GT(bottom[0]->count(), bottom[0]->num()) <<
      "Remove layer data blob must be count > num";
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->num());
  top_shape.push_back(bottom[0]->channels()-1);
  for (int k = 2; k < bottom[0]->num_axes(); k++) {
    top_shape.push_back(bottom[0]->shape(k));
  }
  for (int k = 0; k < bottom[0]->num_axes(); k++) {
    LOG(INFO) << top_shape[k];
  }
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RemoveLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int channel_size = bottom[0]->count(2);
  const int bottom_num_offset = bottom[0]->count(1);
  const int top_num_offset = top[0]->count(1);

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < num; n++) {
    const int remove_idx = bottom[0]->cpu_data()[n];
	const int before_size = channel_size * remove_idx;
	const int after_size = channel_size * (channels - remove_idx - 1);
#ifdef DEBUG
    CHECK_LT(remove_idx, channels) << "Index out of bounds";
#endif

 	// copy channels before the remove idx
	caffe_copy(before_size,
	       bottom_data + n * bottom_num_offset, 
	       top_data + n * top_num_offset);
	// copy channels after the remove idx
	caffe_copy(after_size,
	       bottom_data + n * bottom_num_offset + before_size + channel_size, 
	       top_data + n * top_num_offset + before_size);
  }
}

template <typename Dtype>
void RemoveLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int channel_size = bottom[0]->count(2);
  const int bottom_num_offset = bottom[0]->count(1);
  const int top_num_offset = top[0]->count(1);

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int n = 0; n < num; n++) {
    const int remove_idx = bottom[0]->cpu_data()[n];
	const int before_size = channel_size * remove_idx;
	const int after_size = channel_size * (channels - remove_idx - 1);

 	// copy channels before the remove idx
	caffe_copy(before_size,
	       top_diff + n * top_num_offset, 
	       bottom_diff + n * bottom_num_offset);
	// set the diff for the removed index to 0
	caffe_set(channel_size, (Dtype) 0, bottom_diff + n * bottom_num_offset + before_size);
	// copy channels after the remove idx
	caffe_copy(after_size,
	       top_diff + n * top_num_offset + before_size, 
		   bottom_diff + n * bottom_num_offset + before_size + channel_size);
  }
}

#ifdef CPU_ONLY
STUB_GPU(RemoveLayer);
#endif

INSTANTIATE_CLASS(RemoveLayer);
REGISTER_LAYER_CLASS(Remove);

}  // namespace caffe
