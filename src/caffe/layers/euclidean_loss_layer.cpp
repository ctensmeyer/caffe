#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  normalize_ = this->layer_param_.loss_param().normalize();
  log_output_ = this->layer_param_.loss_param().log_output();
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  const Dtype norm = (normalize_) ? count : bottom[0]->num();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / norm / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  if (log_output_) {
  	int num_instances = bottom[0]->shape(0);
	int num_values = bottom[0]->count(1);
	for (int n = 0; n < num_instances; n++) {
	  ostringstream stream_actual;
	  ostringstream stream_predicted;
	  for (int x = 0; x < num_values; x++) {
	  	int idx = n * num_values + x;
		stream_predicted << bottom[0]->cpu_data()[idx] << " ";
		stream_actual << bottom[1]->cpu_data()[idx] << " ";
	  }
	  LOG(INFO) << "[" << this->layer_param().name() << "] Actual values: " << stream_actual.str() << " Predicted values: " << stream_predicted.str();
	}
  }
  ++count;
}


template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype norm = (normalize_) ? bottom[i]->count() : bottom[i]->num();
	  const Dtype alpha = sign * top[0]->cpu_diff()[0] / norm;
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
