#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/tukey_biweight_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void TukeyBiweightLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  normalize_ = this->layer_param_.loss_param().normalize();
  c_ = this->layer_param_.tukey_biweight_param().c();
  initial_period_ = this->layer_param_.tukey_biweight_param().initial_period();
  initial_mult_ = this->layer_param_.tukey_biweight_param().initial_mult();
  num_iters_ = 0;
  outlier_slope_ = this->layer_param_.tukey_biweight_param().outlier_slope();

  int size = this->layer_param_.tukey_biweight_param().scale_size();
  if (size == 0) {
    scales_.push_back(Dtype(1));
  } else {
    for (int i = 0; i < size; i++) {
	  scales_.push_back(1.4836 * this->layer_param_.tukey_biweight_param().scale(i));
	}
  }
}

template <typename Dtype>
void TukeyBiweightLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
}

template <typename Dtype>
void TukeyBiweightLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  const Dtype norm = (normalize_) ? count : bottom[0]->num();

  const Dtype* data = bottom[0]->cpu_data();
  const Dtype* gt = bottom[1]->cpu_data();

  double const_val = c_ * c_ / 6;
  double loss = 0;
  for (int i = 0; i < count; i++) {
    Dtype r = std::abs(data[i] - gt[i]);
	Dtype scale = scales_[i % scales_.size()];
	if (num_iters_ < initial_period_) {
	  scale *= initial_mult_;
	}
	Dtype r_mad = r / scale;

	if (r_mad < c_) {
	  loss += const_val * (1 - std::pow(1 - std::pow(r_mad / c_, 2),3));
	} else {
	  loss += const_val + (r_mad - c_) * outlier_slope_ ;
	}
  }
  loss /= norm;
  top[0]->mutable_cpu_data()[0] = (Dtype) loss;
}


template <typename Dtype>
void TukeyBiweightLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  const Dtype norm = (normalize_) ? count : bottom[0]->num();

  const Dtype loss_diff = top[0]->cpu_diff()[0];
  const Dtype* data = bottom[0]->cpu_data();
  const Dtype* gt = bottom[1]->cpu_data();
  Dtype* diff = bottom[0]->mutable_cpu_diff();

  if (propagate_down[0]) {
    for (int i = 0; i < count; i++) {
      Dtype r = data[i] - gt[i];
	  Dtype scale = scales_[i % scales_.size()];
	  if (num_iters_ < initial_period_) {
	    scale *= initial_mult_;
	  }
	  Dtype r_mad = r / scale;

	  Dtype val = 0;
	  if (std::abs(r_mad) < c_) {
	    val = loss_diff * r / scale / scale * std::pow((1 - std::pow(r / c_ / scale, 2)), 2) / norm;
	  } else {
	    if (r_mad >= 0) {
	      val = loss_diff * outlier_slope_ / scale / norm;
		} else {
		  val = -1 * loss_diff * outlier_slope_ / scale / norm;
		}
	  }
	  diff[i] = val;
	}
  }
  num_iters_++;
}


INSTANTIATE_CLASS(TukeyBiweightLossLayer);
REGISTER_LAYER_CLASS(TukeyBiweightLoss);

}  // namespace caffe
