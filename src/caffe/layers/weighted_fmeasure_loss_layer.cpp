#include <algorithm>
#include <cfloat>
#include <vector>
#include <math.h>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  margin_ = this->layer_param().weighted_fmeasure_loss_param().margin();
  mse_lambda_ = this->layer_param().weighted_fmeasure_loss_param().mse_lambda();

  if (!this->layer_param().weighted_fmeasure_loss_param().avg_fm()) {
    CHECK_EQ(top.size(), 1) << "With Total FM, only 1 top blob is supported";
  }
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

  if (this->layer_param().weighted_fmeasure_loss_param().avg_fm()) {
    vector<int> shape2;
    shape2.push_back(1);
    shape2.push_back(2);
    shape2.push_back(bottom[0]->height());
    shape2.push_back(bottom[0]->width());
    work_buffer_->Reshape(shape2);
  } else {
    work_buffer_->Reshape(bottom[0]->shape());
  }
}

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param().weighted_fmeasure_loss_param().avg_fm()) {
    Forward_cpu_avg_fm(bottom, top);
  } else {
    Forward_cpu_total_fm(bottom, top);
  }
}

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Forward_cpu_total_fm(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
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

  // threshold inputs according to margin
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

  // Blas version
  Dtype* target_mult_input = work_buffer_->mutable_cpu_data();
  caffe_mul(count, input, target, target_mult_input);

  recall_num[0] = caffe_cpu_dot(count, target_mult_input, recall_weight);
  recall_denum[0] = caffe_cpu_dot(count, target, recall_weight);

  precision_num[0] = caffe_cpu_dot(count, target_mult_input, precision_weight);
  precision_denum[0] = caffe_cpu_dot(count, input, precision_weight);

  recall[0] = (recall_denum[0] != 0) ? recall_num[0] / recall_denum[0] : 0;
  precision[0] = (precision_denum[0] != 0) ? precision_num[0] / precision_denum[0] : 0;

  if (precision[0] != 0 && recall[0] != 0) {
    Dtype f_measure = 2 * recall[0] * precision[0] / (recall[0] + precision[0]);
    top[0]->mutable_cpu_data()[0] = 1 - f_measure;
  } else {
    // compute MSE when f-measure is undefined
    Dtype* work = work_buffer_->mutable_cpu_data();
    Dtype* work2 = work_buffer_->mutable_cpu_diff();
    caffe_sub(count, target, input, work);

    caffe_mul(count, work, work, work2);
    caffe_mul(count, work2, precision_weight, work);

    Dtype numer = caffe_cpu_asum(count, work);
    Dtype denum = caffe_cpu_asum(count, precision_weight);
    Dtype loss = mse_lambda_ * 0.5 * numer / denum;
    //LOG(INFO) << "HERE: " << numer << " " << denum << " " << (numer / denum) << " " << loss;
    top[0]->mutable_cpu_data()[0] = loss;
    if (!std::isfinite(loss)) {
      LOG(INFO) << "Found non-finite MSE: " << loss;
      LOG(INFO) << "\tLayer name: " << this->layer_param().name();
      LOG(INFO) << "\tnumer: " << numer << " denum: " << denum;
    } 
    
/*
    numer = 0;
    denum = 0;
    for (int i = 0; i < count; i++) {
      numer += (target[i] - input[i]) * (target[i] - input[i]) * precision_weight[i];
      denum += precision_weight[i];
    }
    loss = mse_lambda_ * 0.5 * numer / denum;
    LOG(INFO) << "HERE2: " << numer << " " << denum << " " << (numer / denum) << " " << loss;
*/
  }

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
}

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Forward_cpu_avg_fm(
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
	if (precision[n] != 0 && recall[n] != 0) {
      pfm[n] = 2 * recall[n] * precision[n] / (recall[n] + precision[n]);
      if (top.size() > 1) {
        top[1]->mutable_cpu_data()[n] = pfm[n];
      }
	  if (!std::isfinite(pfm[n])) {
	    LOG(INFO) << "Found " << pfm[n] << " which is non-finite on instance " << n;
	    LOG(INFO) << "\tLayer name: " << this->layer_param().name();
	    LOG(INFO) << "\tprecision: " << precision[n] << " precision_num: " 
			<< precision_num[n] << " precision_denum: " << precision_denum[n];
	    LOG(INFO) << "\trecall: " << recall[n] << " recall_num: " 
			<< recall_num[n] << " recall_denum: " << recall_denum[n];
		for (int i = 0; i < spatial_size; i++) {
		  if (!std::isfinite(input[i])) {
		    LOG(INFO) << "\tinput at (" << (i / width) << "," << (i % width) 
				<< ") is " << input[i] << " which is non-finite";
		  }
		  if (!std::isfinite(target[i])) {
		    LOG(INFO) << "\ttarget at (" << (i / width) << "," << (i % width) 
				<< ") is " << target[i] << " which is non-finite";
		  }
		  if (!std::isfinite(precision_weight[i])) {
		    LOG(INFO) << "\tprecision_weight at (" << (i / width) << "," << (i % width) 
				<< ") is " << precision_weight[i] << " which is non-finite";
		  }
		  if (!std::isfinite(recall_weight[i])) {
		    LOG(INFO) << "\trecall_weight at (" << (i / width) << "," << (i % width) 
				<< ") is " << recall_weight[i] << " which is non-finite";
		  }
		}
	  }
	  loss += 1 - pfm[n];
	} else {
	  // GT is likely all background, so Recall is undefined and Precision is 0
	  // Recall Weights don't apply, so do \sum W_p (B - G)^2 / \sum W_p
      Dtype* work = work_buffer_->mutable_cpu_data();
      caffe_sub(spatial_size, target + spatial_offset, input + spatial_offset, work);

	  caffe_mul(spatial_size, work, work, work + spatial_size);
	  caffe_mul(spatial_size, work + spatial_size, precision_weight + spatial_offset, work);

	  Dtype numer = caffe_cpu_asum(spatial_size, work);
	  Dtype denum = caffe_cpu_asum(spatial_size, precision_weight + spatial_offset);
	  //LOG(INFO) << "HERE: " << numer << " " << denum << " " << (numer / denum) << " " << loss;
	  Dtype individual_loss = mse_lambda_ * 0.5 * numer / denum;
	  if (!std::isfinite(individual_loss)) {
	    LOG(INFO) << "Found non-finite MSE: " << individual_loss << " on instance " << n;
	    LOG(INFO) << "\tLayer name: " << this->layer_param().name();
	    LOG(INFO) << "\tnumer: " << numer << " denum: " << denum;
	  }
	  loss += individual_loss;
	  //LOG(INFO) << loss;
	}
  }
  top[0]->mutable_cpu_data()[0] = loss / num;


}

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1] || propagate_down[2] || propagate_down[3]) {
    LOG(FATAL) << this->type()
               << " WeightedFmeasureLossLayer cannot backpropagate to label inputs, or weight maps.";
  }
  if (!propagate_down[0]) {
    return;
  }

  if (this->layer_param().weighted_fmeasure_loss_param().avg_fm()) {
    Backward_cpu_avg_fm(top, propagate_down, bottom);
  } else {
    Backward_cpu_total_fm(top, propagate_down, bottom);
  }
}

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Backward_cpu_total_fm(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* input = bottom[0]->cpu_data(); 
  const Dtype* target = bottom[1]->cpu_data(); 
  const Dtype* recall_weight = bottom[2]->cpu_data();
  const Dtype* precision_weight = bottom[3]->cpu_data();
  Dtype* diff = bottom[0]->mutable_cpu_diff();
  const Dtype top_diff = top[0]->cpu_diff()[0];
  const int count = bottom[0]->count();


  // cache values for backward step
  const Dtype* recall = recall_->cpu_data();
  //const Dtype* recall_num = recall_num_->cpu_data();  // not need for backwards step
  const Dtype* recall_denum = recall_denum_->cpu_data();
  const Dtype* precision = precision_->cpu_data();
  const Dtype* precision_num = precision_num_->cpu_data();
  const Dtype* precision_denum = precision_denum_->cpu_data();
  
  // need to compute dF/dB = dF/dR * dR/dB + dF/dP * dP/dB for each pixel
  // dF/dR and dF/dP are fixed for all pixels
  // dF/dR = 2 * p^2 / (p + r)^2
  // dF/dP = 2 * r^2 / (p + r)^2
  //DLOG(ERROR) << "-F/P/R:" << top[0]->cpu_data()[0] << " " << precision_ << " " <<  recall_;
  //DLOG(ERROR) << "P_num/P_denum:" << precision_num_ << " " << precision_denum_;
  //DLOG(ERROR) << "R_num/R_denum:" << recall_num_ << " " << recall_denum_;
  if (precision[0] != 0 && recall[0] != 0) {
    Dtype sum_squared = recall[0] + precision[0];
    sum_squared = sum_squared * sum_squared;
    Dtype dF_dR = 2 * precision[0] * precision[0] / sum_squared; 
    Dtype dF_dP = 2 * recall[0] * recall[0] / sum_squared;
    //DLOG(ERROR) << "dF_dR/dF_dP:" << dF_dR << " " << dF_dP;
  
    // BLAS Version, overwritting input buffers for space saving
    // dF_dR * dR_dB  
    Dtype* dR_dB = work_buffer_->mutable_cpu_data();
    caffe_mul(count, target, recall_weight, dR_dB);
    caffe_scal(count, (Dtype)(-2. * dF_dR / recall_denum[0]), dR_dB); 
  
    // dF_dP * dP_dB 
    Dtype* dP_dB = work_buffer_->mutable_cpu_diff();
    caffe_copy(count, target, dP_dB);
    caffe_scal(count, precision_denum[0], dP_dB); // scale target by precision_denum_
    caffe_add_scalar(count, -1 * precision_num[0], dP_dB); // subtract precision_num_
    caffe_mul(count, dP_dB, precision_weight, dP_dB);
    caffe_scal(count, (Dtype)(-1 * top_diff * dF_dP / (precision_denum[0] * precision_denum[0])), dP_dB);
  
    caffe_add(count, dR_dB, dP_dB, diff);
  } else {
	LOG(INFO) << "BACK";
    Dtype* work = work_buffer_->mutable_cpu_data();
    Dtype* work2 = work_buffer_->mutable_cpu_diff();
    caffe_set(count, (Dtype)0., work);
    caffe_set(count, (Dtype)0., work2);

    caffe_sub(count, input, target, work);

	caffe_mul(count, work, precision_weight, work2);
	Dtype denum = caffe_cpu_asum(count, precision_weight);
	caffe_scal(count, (Dtype) (top_diff * mse_lambda_ / denum), work2);

    caffe_add(count, diff, work2, diff);
/*
	for (int i = 0; i < count; i++) {
       Dtype loop_diff = top_diff * mse_lambda_ * precision_weight[i] *  (input[i] - target[i]) / denum; 
	   LOG(INFO) << "idx i: " << i << " Loop diff: " << loop_diff << " Blas diff: " << diff[i];
    }
*/
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

template <typename Dtype>
void WeightedFmeasureLossLayer<Dtype>::Backward_cpu_avg_fm(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* input = bottom[0]->cpu_data();  
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* recall_weight = bottom[2]->cpu_data();
  const Dtype* precision_weight = bottom[3]->cpu_data();
  const Dtype top_diff = top[0]->cpu_diff()[0];

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
      caffe_scal(spatial_size, (Dtype)(-1 * top_diff * dF_dP / (precision_denum[n] * precision_denum[n])), dP_dB);

      caffe_add(spatial_size, dR_dB, dP_dB, diff + spatial_offset);
    } else {
      Dtype* work = work_buffer_->mutable_cpu_data();
      caffe_sub(spatial_size, input + spatial_offset, target + spatial_offset, work + spatial_size);

	  caffe_mul(spatial_size, work + spatial_size, precision_weight + spatial_offset, work);
	  Dtype denum = caffe_cpu_asum(spatial_size, precision_weight + spatial_offset);
	  caffe_scal(spatial_size, (Dtype) (mse_lambda_ * top_diff / denum), work);

      caffe_add(spatial_size, diff + spatial_offset, work, diff + spatial_offset);
    }
  }
  caffe_scal(count, (Dtype) (1. / num), diff);
}

INSTANTIATE_CLASS(WeightedFmeasureLossLayer);
REGISTER_LAYER_CLASS(WeightedFmeasureLoss);

}  // namespace caffe
