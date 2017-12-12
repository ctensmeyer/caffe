#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxFullLoss2Layer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void SoftmaxFullLoss2Layer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);  // number of soft maxes to compute
  inner_num_ = bottom[0]->count(softmax_axis_);  // number of elements in each softmax
  CHECK_EQ(outer_num_, bottom[1]->count(0, softmax_axis_)) << "sizes must match";
  CHECK_EQ(inner_num_, bottom[1]->count(softmax_axis_)) << "sizes must match";

  vector<int> shape;
  shape.push_back(outer_num_);
  top[0]->Reshape(shape);  // outputs loss of individual softmaxes
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxFullLoss2Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* target_prob_data = bottom[1]->cpu_data();
  for (int i = 0; i < outer_num_; i++) {
    Dtype loss = 0;
    for (int j = 0; j < inner_num_; j++) {
	  int idx = i * inner_num_ + j;
      loss -= target_prob_data[idx] * 
        			log(std::max(prob_data[idx], Dtype(FLT_MIN)));
    }
    if (normalize_) {
      top[0]->mutable_cpu_data()[i] = loss / bottom[0]->count();
    } else {
      top[0]->mutable_cpu_data()[i] = loss / outer_num_;
    }
  }

  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxFullLoss2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* target_prob_data = bottom[1]->cpu_data();

    for (int i = 0; i < outer_num_; i++) {
      const Dtype loss_weight = top[0]->cpu_diff()[i];
	  Dtype mult = 0;
      if (normalize_) {
	    mult = loss_weight / bottom[0]->count();
      } else {
	    mult = loss_weight / outer_num_;
      }
      for (int j = 0; j < inner_num_; j++) {
	    int idx = i * inner_num_ + j;
        bottom_diff[idx] = mult * (prob_data[idx] - target_prob_data[idx]);
	  }
	}
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxFullLoss2Layer);
#endif

INSTANTIATE_CLASS(SoftmaxFullLoss2Layer);
REGISTER_LAYER_CLASS(SoftmaxFullLoss2);

}  // namespace caffe
