#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layers/sigmoid_ce_loss_layer2.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLoss2Layer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  normalize_ = this->layer_param_.loss_param().normalize(); 
}

template <typename Dtype>
void SigmoidCrossEntropyLoss2Layer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);

  vector<int> shape;
  shape.push_back(bottom[0]->num());
  shape.push_back(1);
  top[0]->Reshape(shape);  // outputs loss for each instance
}

template <typename Dtype>
void SigmoidCrossEntropyLoss2Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = count / num;
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();

  for (int n = 0; n < num; n++) {
    Dtype instance_loss = 0;

    for (int i = 0; i < dim; ++i) {
	  int idx = n * dim + i;
      Dtype loss = input_data[idx] * (target[idx] - (input_data[idx] >= 0)) -
          log(1 + exp(input_data[idx] - 2 * input_data[idx] * (input_data[idx] >= 0)));
      instance_loss -= loss;
    }

    if (normalize_) {
	  instance_loss /= count;
    } else {
	  instance_loss /= num;
	}
    top[0]->mutable_cpu_data()[n] = instance_loss;
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLoss2Layer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int dim = count / num;
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);

    for (int n = 0; n < num; n++) {
      // Scale down gradient
      Dtype loss_weight = top[0]->cpu_diff()[n];
	  if (normalize_) {
	    loss_weight /= count;
	  } else {
	    loss_weight /= num;
      }
      caffe_scal(dim, loss_weight, bottom_diff + n * dim);
	}
  }
}


INSTANTIATE_CLASS(SigmoidCrossEntropyLoss2Layer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss2);

}  // namespace caffe
