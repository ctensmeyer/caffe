#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxFullLossLayer<Dtype>::LayerSetUp(
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
void SoftmaxFullLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);  // number of soft maxes to compute
  inner_num_ = bottom[0]->count(softmax_axis_);  // number of elements in each softmax
  CHECK_EQ(outer_num_, bottom[1]->count(0, softmax_axis_)) << "sizes must match";
  CHECK_EQ(inner_num_, bottom[1]->count(softmax_axis_)) << "sizes must match";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxFullLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* target_prob_data = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < prob_.count(); ++i) {
	  loss -= target_prob_data[i] * 
	  			log(std::max(prob_data[i], Dtype(FLT_MIN)));
  }

  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / bottom[0]->count();
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxFullLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    //caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* target_prob_data = bottom[1]->cpu_data();
    for (int i = 0; i < prob_.count(); ++i) {
        bottom_diff[i] = prob_data[i] - target_prob_data[i];
	}

    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / bottom[0]->count(), bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxFullLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxFullLossLayer);
REGISTER_LAYER_CLASS(SoftmaxFullLoss);

}  // namespace caffe
