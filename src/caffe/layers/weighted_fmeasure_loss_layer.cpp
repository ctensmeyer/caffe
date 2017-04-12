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
  margin_ = this->layer_param().weighted_fmeasure_loss_param().margin();
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

  vector<int> shape;
  shape.push_back(bottom[0]->num());
  shape.push_back(1);
  pfm_->Reshape(shape);
  recall_->Reshape(shape);
  recall_num_->Reshape(shape);
  recall_denum_->Reshape(shape);
  precision_->Reshape(shape);
  precision_num_->Reshape(shape);
  precision_denum_->Reshape(shape);
  if (top.size() > 1) {
    top[1]->Reshape(shape);
  }

  vector<int> shape2;
  shape2.push_back(1);
  shape2.push_back(2);
  shape2.push_back(bottom[0]->height());
  shape2.push_back(bottom[0]->width());

  work_buffer_->Reshape(shape2);
}

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // constants
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int spatial_size = height * width;

  // inputs
  Dtype* input = bottom[0]->mutable_cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* recall_weight = bottom[2]->cpu_data();
  const Dtype* precision_weight = bottom[3]->cpu_data();

  // cache values for backward step
  Dtype* recall = recall_->mutable_cpu_data();
  Dtype* recall_num = recall_num_->mutable_cpu_data();
  Dtype* recall_denum = recall_denum_->mutable_cpu_data();
  Dtype* precision = precision_->mutable_cpu_data();
  Dtype* precision_num = precision_num_->mutable_cpu_data();
  Dtype* precision_denum = precision_denum_->mutable_cpu_data();

  Dtype* pfm = pfm_->mutable_cpu_data();
  Dtype loss = 0;

  // threshold predictions according to margin
  if (margin_ > 0 && margin_ < 0.5) {
    for (int i = 0; i < count; i++) {
	  if (target[i] > 0.5) {
		if (input[i] >= (0.5 + margin_)) {
		  input[i] = target[i];
		}
	  } else {
		if (input[i] <= (0.5 - margin_)) {
		  input[i] = target[i];
		}
	  }
	}
  }

  for (int n = 0; n < num; n++) {
    const int spatial_offset = n * spatial_size;

    Dtype* target_mult_input = work_buffer_->mutable_cpu_data();
    caffe_mul(spatial_size, input + spatial_offset, target + spatial_offset, target_mult_input);

    recall_num[n] = caffe_cpu_dot(spatial_size, target_mult_input, recall_weight + spatial_offset);
    recall_denum[n] = caffe_cpu_dot(spatial_size, target + spatial_offset, recall_weight + spatial_offset);

    precision_num[n] = caffe_cpu_dot(spatial_size, target_mult_input, precision_weight + spatial_offset);
    precision_denum[n] = caffe_cpu_dot(spatial_size, input + spatial_offset, precision_weight + spatial_offset);

    // check for divide by zero errors
    if (recall_denum[n] != 0) {
	  recall[n] = recall_num[n] / recall_denum[n];
	} else {
	  recall[n] = 0;
	}
    if (precision_denum[n] != 0) {
	  precision[n] = precision_num[n] / precision_denum[n];
	} else {
	  precision[n] = 0;
	}
	if (precision[n] + recall[n] != 0) {
      pfm[n] = 2 * recall[n] * precision[n] / (recall[n] + precision[n]);
      if (top.size() > 1) {
        top[1]->mutable_cpu_data()[n] = pfm[n];
      }
	  loss += 1 - pfm[n];
	} else {
	  // GT is all background, so Recall is undefined and Precision is 0
	  // Recall Weights don't apply, so do \sum W_p (B - G)^2 / \sum W_p
	  // and G == 0
      Dtype* work = work_buffer_->mutable_cpu_data();
	  caffe_copy(spatial_size, input + spatial_offset, work);

	  caffe_mul(spatial_size, input + spatial_offset, work, work + spatial_size);
	  caffe_mul(spatial_size, work + spatial_size, precision_weight + spatial_offset, work);

	  Dtype numer = caffe_cpu_asum(spatial_size, work);
	  Dtype denum = caffe_cpu_asum(spatial_size, precision_weight + spatial_offset);
	  //LOG(INFO) << "HERE: " << numer << " " << denum << " " << (numer / denum) << " " << loss;
	  loss += 0.5 * numer / denum;
	  //LOG(INFO) << loss;
	}
    // check for 0 denominators to avoid nans
    //LOG(INFO) << "F/P/R:" << f_measure << " " << precision_ << " " <<  recall_;
    //LOG(INFO) << "P_num/P_denum:" << precision_num_ << " " << precision_denum_;
    //LOG(INFO) << "R_num/R_denum:" << recall_num_ << " " << recall_denum_;
  }
  top[0]->mutable_cpu_data()[0] = loss / num;

/*
  // Loop version
  // not updated to compute the average of individual p-fms
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

}

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1] || propagate_down[2] || propagate_down[3]) {
    LOG(FATAL) << this->type()
               << " WeightedFmeasureLossLayer cannot backpropagate to label inputs, or weight maps.";
  }
  
  const Dtype* input = bottom[0]->cpu_data();  
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* recall_weight = bottom[2]->cpu_data();
  const Dtype* precision_weight = bottom[3]->cpu_data();

  // constants
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int spatial_size = height * width;

  // cached values from the forward step
  const Dtype* recall = recall_->cpu_data();
  //const Dtype* recall_num = recall_num_->cpu_data();
  const Dtype* recall_denum = recall_denum_->cpu_data();
  const Dtype* precision = precision_->cpu_data();
  const Dtype* precision_num = precision_num_->cpu_data();
  const Dtype* precision_denum = precision_denum_->cpu_data();

  Dtype* diff = bottom[0]->mutable_cpu_diff();

  if (!propagate_down[0]) {
    return;
  }

  // need to compute dF/dB = dF/dR * dR/dB + dF/dP * dP/dB for each pixel
  // dF/dR and dF/dP are fixed for all pixels
  // dF/dR = 2 * p^2 / (p + r)^2
  // dF/dP = 2 * r^2 / (p + r)^2
  for(int n = 0; n < num; n++) {
    const int spatial_offset = n * spatial_size;
  
    //DLOG(ERROR) << "-F/P/R:" << top[0]->cpu_data()[0] << " " << precision_ << " " <<  recall_;
    if (precision[n] != 0 && recall[n] != 0) {
      Dtype sum_squared = recall[n] + precision[n];
      sum_squared = sum_squared * sum_squared;
      Dtype dF_dR = 2 * precision[n] * precision[n] / sum_squared; 
      Dtype dF_dP = 2 * recall[n] * recall[n] / sum_squared;
      //DLOG(ERROR) << "dF_dR/dF_dP:" << dF_dR << " " << dF_dP;
  
      // BLAS Version, overwritting input buffers for space saving
      // dF_dR * dR_dB  
      Dtype* dR_dB = work_buffer_->mutable_cpu_data();
      caffe_mul(spatial_size, target + spatial_offset, recall_weight + spatial_offset, dR_dB);
      caffe_scal(spatial_size, (Dtype)(-2. * dF_dR / recall_denum[n]), dR_dB); 
  
      // dF_dP * dP_dB 
      Dtype* dP_dB = work_buffer_->mutable_cpu_diff();
      caffe_copy(spatial_size, target + spatial_offset, dP_dB);
      caffe_scal(spatial_size, precision_denum[n], dP_dB); // scale target by precision_denum_
      caffe_add_scalar(spatial_size, -1 * precision_num[n], dP_dB); // subtract precision_num_
      caffe_mul(spatial_size, dP_dB, precision_weight + spatial_offset, dP_dB);
      caffe_scal(spatial_size, (Dtype)(-2. * dF_dP / (precision_denum[n] * precision_denum[n])), dP_dB);

      caffe_add(spatial_size, dR_dB, dP_dB, diff + spatial_offset);
    } else {
      Dtype* work = work_buffer_->mutable_cpu_data();

	  caffe_mul(spatial_size, input + spatial_offset, precision_weight + spatial_offset, work);
	  Dtype denum = caffe_cpu_asum(spatial_size, precision_weight + spatial_offset);
	  caffe_scal(spatial_size, (Dtype) (2. / denum), work);

      caffe_add(spatial_size, diff + spatial_offset, work, diff + spatial_offset);
    }
/*
	// LOOP Version. Out of date
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
  caffe_scal(count, (Dtype) (1. / num), diff);
}

//#ifdef CPU_ONLY
//STUB_GPU(WeightedFmeasureLossLayer);
//#endif

INSTANTIATE_CLASS(WeightedFmeasureLossLayer);
REGISTER_LAYER_CLASS(WeightedFmeasureLoss);

}  // namespace caffe
