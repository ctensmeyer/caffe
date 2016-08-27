#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "WeightedFmeasure layer inputs must have the same count.";
  CHECK_EQ(bottom[0]->count(), bottom[2]->count()) <<
      "WeightedFmeasure layer inputs must have the same count.";
  CHECK_EQ(bottom[0]->count(), bottom[3]->count()) <<
      "WeightedFmeasure layer inputs must have the same count.";
  CHECK_EQ(bottom[0]->num(), bottom[2]->num()) <<
      "WeightedFmeasure layer inputs must have the same number of instances.";
  CHECK_EQ(bottom[0]->num(), bottom[3]->num()) <<
      "WeightedFmeasure layer inputs must have the same number of instances.";
  work_buffer_->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  // Stable version of loss computation from input data
  const Dtype* input = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* recall_weight = bottom[2]->cpu_data();
  const Dtype* precision_weight = bottom[3]->cpu_data();

  recall_num_ = recall_denum_ = precision_num_ = precision_denum_ = 0;

  // Blas version
  Dtype* target_mult_input = work_buffer_->mutable_cpu_data();
  caffe_mul(count, input, target, target_mult_input);

  recall_num_ = caffe_cpu_dot(count, target_mult_input, recall_weight);
  recall_denum_ = caffe_cpu_dot(count, target, recall_weight);

  precision_num_ = caffe_cpu_dot(count, target_mult_input, precision_weight);
  precision_denum_ = caffe_cpu_dot(count, input, precision_weight);

/*
  // Loop version
  for (int i = 0; i < count; ++i) {
    // Equation 1 from Performance evaluation methodology for 
	// historical document image binarization
	// B(x,y) is the binarized image under evaluation.  In this work, it assumes continous
	// probabilities of being foreground.  G(x,y) is the binary (1 for foreground, 0 for background)
	// reference image used to score B(x,y).  G_W(x,y) is the weighted binary image, G(x,y) * R(x,y)
	// where R(x,y) are the precomputed Recall Weights
    recall_num_ += input[i] * (target[i] * recall_weight[i]);  // B(x,y) * (G_W(x,y))
	recall_denum_ += (target[i] * recall_weight[i]);  // G_W(x,y)

    // Equation 9
	// G(x,y) as above.  B_W(x,y) is the weighted binarized image, B(x,y) * P(x,y), where P(x,y)
	// are the precomputed Precision Weights
	precision_num_ += target[i] * (input[i] * precision_weight[i]);  // G(x,y) * B_W(x,y)
	precision_denum_ += (input[i] * precision_weight[i]);  // B_W(x,y)
  }
*/

  // check for 0 denominators to avoid nans
  recall_ = (recall_denum_ != 0) ? recall_num_ / recall_denum_ : 0;
  precision_ = (precision_denum_ != 0) ? precision_num_ / precision_denum_ : 0;
  Dtype f_measure = ((recall_ + precision_) != 0) ? 2 * recall_ * precision_ / (recall_ + precision_) : 0;
  //LOG(INFO) << "F/P/R:" << f_measure << " " << precision_ << " " <<  recall_;
  //LOG(INFO) << "P_num/P_denum:" << precision_num_ << " " << precision_denum_;
  //LOG(INFO) << "R_num/R_denum:" << recall_num_ << " " << recall_denum_;
  top[0]->mutable_cpu_data()[0] = 1 - f_measure;  // loss should be lower is better
}

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1] || propagate_down[2] || propagate_down[3]) {
    LOG(FATAL) << this->type()
               << " WeightedFmeasureLossLayer cannot backpropagate to label inputs, or weight maps.";
  }
  if (propagate_down[0]) {
    //const Dtype* input = bottom[0]->cpu_data();
    Dtype* target = bottom[1]->mutable_cpu_data();  // reuse memory for intermediate computations
    const Dtype* recall_weight = bottom[2]->cpu_data();
    const Dtype* precision_weight = bottom[3]->cpu_data();
	Dtype* diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();

	// need to compute dF/dB = dF/dR * dR/dB + dF/dP * dP/dB for each pixel
	// dF/dR and dF/dP are fixed for all pixels
	// dF/dR = 2 * p^2 / (p + r)^2
	// dF/dP = 2 * r^2 / (p + r)^2
    //DLOG(ERROR) << "-F/P/R:" << top[0]->cpu_data()[0] << " " << precision_ << " " <<  recall_;
    //DLOG(ERROR) << "P_num/P_denum:" << precision_num_ << " " << precision_denum_;
    //DLOG(ERROR) << "R_num/R_denum:" << recall_num_ << " " << recall_denum_;
	if (precision_ != 0 && recall_ != 0) {
	  Dtype sum_squared = recall_ + precision_;
	  sum_squared = sum_squared * sum_squared;
	  Dtype dF_dR = 2 * precision_ * precision_ / sum_squared; 
	  Dtype dF_dP = 2 * recall_ * recall_ / sum_squared;
      //DLOG(ERROR) << "dF_dR/dF_dP:" << dF_dR << " " << dF_dP;

	  // BLAS Version, overwritting input buffers for space saving
	  // dF_dR * dR_dB  
	  Dtype* dR_dB = work_buffer_->mutable_cpu_data();
	  caffe_mul(count, target, recall_weight, dR_dB);
	  caffe_scal(count, (Dtype)(-2. * dF_dR / recall_denum_), dR_dB); 

	  // dF_dP * dP_dB 
	  Dtype* dP_dB = work_buffer_->mutable_cpu_diff();
	  caffe_copy(count, target, dP_dB);
	  caffe_scal(count, precision_denum_, dP_dB); // scale target by precision_denum_
	  caffe_add_scalar(count, -1 * precision_num_, dP_dB); // subtract precision_num_
	  caffe_mul(count, dP_dB, precision_weight, dP_dB);
	  caffe_scal(count, (Dtype)(-2. * dF_dP / (precision_denum_ * precision_denum_)), dP_dB);

	  caffe_add(count, dR_dB, dP_dB, diff);
	} else {
	  // fmeasure is actually undefined in this case, so just leave gradient = 0
	}

/*
	// LOOP Version
	// computing dR/dB and dP/dB
	Dtype dFloss_dF = -1;
    for (int i = 0; i < count; ++i) {
	  Dtype dR_dB = target[i] * recall_weight[i] / recall_denum_;  // this one is simple

	  Dtype dP_dB = precision_weight[i] * (precision_denum_ * target[i] - precision_num_) / 
	  					(precision_denum_ * precision_denum_);
	  diff[i] = 2 * dFloss_dF * (dF_dR * dR_dB + dF_dP * dP_dB);
	  //diff[i] = 2 * dFloss_dF * (dF_dR * dR_dB[i] + dF_dP * dP_dB);
      //DLOG(ERROR) << "i/dR_dB/dP_dB/diff:" << i << " " << dR_dB << " " << dP_dB << " " << diff[i];
	}
*/
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedFmeasureLossLayer);
#endif

INSTANTIATE_CLASS(WeightedFmeasureLossLayer);
REGISTER_LAYER_CLASS(WeightedFmeasureLoss);

}  // namespace caffe
