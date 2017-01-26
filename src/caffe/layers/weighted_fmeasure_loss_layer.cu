#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
__global__ void MarginThreshold(const int n, const Dtype margin,
    Dtype* input, const Dtype* target) {
  CUDA_KERNEL_LOOP(index, n) {
    if (target[index] > 0.5) {
      if (input[index] >= (1. - margin)) {
        input[index] = target[index];
      }
    } else {
      if (input[index] <= margin) {
        input[index] = target[index];
      }
    }
  }
}

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  // Stable version of loss computation from input data
  Dtype* input = bottom[0]->mutable_gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  const Dtype* recall_weight = bottom[2]->gpu_data();
  const Dtype* precision_weight = bottom[3]->gpu_data();

  recall_num_ = recall_denum_ = precision_num_ = precision_denum_ = 0;

  // threshold inputs according to margin
  if (margin_ > 0 && margin_ < 0.5) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    MarginThreshold<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, margin_, input, target);
    CUDA_POST_KERNEL_CHECK;
  }

  // Blas version
  Dtype* target_mult_input = work_buffer_->mutable_gpu_data();
  caffe_gpu_mul(count, input, target, target_mult_input);

  caffe_gpu_dot(count, target_mult_input, recall_weight, &recall_num_);
  caffe_gpu_dot(count, target, recall_weight, &recall_denum_);

  caffe_gpu_dot(count, target_mult_input, precision_weight, &precision_num_);
  caffe_gpu_dot(count, input, precision_weight, &precision_denum_);

  // check for 0 denominators to avoid nans
  recall_ = (recall_denum_ != 0) ? recall_num_ / recall_denum_ : 0;
  precision_ = (precision_denum_ != 0) ? precision_num_ / precision_denum_ : 0;
  Dtype f_measure = ((recall_ + precision_) != 0) ? 2 * recall_ * precision_ / (recall_ + precision_) : 0;
  top[0]->mutable_cpu_data()[0] = 1 - f_measure;  // loss should be lower is better
}

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1] || propagate_down[2] || propagate_down[3]) {
    LOG(FATAL) << this->type()
               << " WeightedFmeasureLossLayer cannot backpropagate to label inputs, or weight maps.";
  }
  if (propagate_down[0]) {
    Dtype* target = bottom[1]->mutable_gpu_data();  // reuse memory for intermediate computations
    const Dtype* recall_weight = bottom[2]->gpu_data();
    const Dtype* precision_weight = bottom[3]->gpu_data();
	Dtype* diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();

	// need to compute dF/dB = dF/dR * dR/dB + dF/dP * dP/dB for each pixel
	// dF/dR and dF/dP are fixed for all pixels
	// dF/dR = 2 * p^2 / (p + r)^2
	// dF/dP = 2 * r^2 / (p + r)^2
	if (precision_ != 0 && recall_ != 0) {
	  Dtype sum_squared = recall_ + precision_;
	  sum_squared = sum_squared * sum_squared;
	  Dtype dF_dR = 2 * precision_ * precision_ / sum_squared; 
	  Dtype dF_dP = 2 * recall_ * recall_ / sum_squared;

	  // dF_dR * dR_dB  
	  Dtype* dR_dB = work_buffer_->mutable_gpu_data();
	  caffe_gpu_mul(count, target, recall_weight, dR_dB);
	  caffe_gpu_scal(count, (Dtype)(-2. * dF_dR / recall_denum_), dR_dB); 

	  // dF_dP * dP_dB 
	  Dtype* dP_dB = work_buffer_->mutable_gpu_diff();
	  caffe_gpu_memcpy(sizeof(Dtype) * count, target, dP_dB);
	  caffe_gpu_scal(count, precision_denum_, dP_dB); // scale target by precision_denum_
	  caffe_gpu_add_scalar(count, -1 * precision_num_, dP_dB); // subtract precision_num_
	  caffe_gpu_mul(count, dP_dB, precision_weight, dP_dB);
	  caffe_gpu_scal(count, (Dtype)(-2. * dF_dP / (precision_denum_ * precision_denum_)), dP_dB);

	  caffe_gpu_add(count, dR_dB, dP_dB, diff);
	} else {
	  // fmeasure is actually undefined in this case, so just leave gradient = 0
	}
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightedFmeasureLossLayer);

}  // namespace caffe
