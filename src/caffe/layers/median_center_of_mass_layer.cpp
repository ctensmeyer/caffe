// Copyright 2017 Chris Tensmeyer

#include <iostream>  // NOLINT(readability/streams)
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MedianCenterOfMassLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MedianCenterOfMassLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  CHECK_EQ(num_axes, 4) << "num_axes must be 4, not " << num_axes;

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  vector<int> shape;
  shape.push_back(num);
  shape.push_back(channels);
  shape.push_back(2);
  top[0]->Reshape(shape);

}

template <typename Dtype>
Dtype find_median_position(const vector<Dtype> cum_vals, int& lower_idx_out, 
                           int& upper_idx_out, Dtype& slope_out) {
  //for (int i = 0; i < cum_vals.size(); i++) {
  //  LOG(INFO) << i << ": " << cum_vals[i];
  //}
  int upper_median_idx = 0;
  Dtype total = cum_vals[cum_vals.size()-1];
  for (int i = 0; i < cum_vals.size(); i++) {
	Dtype cum_val = cum_vals[i];
	Dtype cum_perc = cum_val / total;
	if (cum_perc > 0.5) {
	  upper_median_idx = i;
	  break;
	}
  }
  // upper_median_idx is now pointing to the first val with cumulative sum
  // greater than 50%.  Now we need to find the first position with a non-zero
  // entry, if any.  cum_vals has cumulative values, so we need to check for
  // zero diffs.  If there aren't any non-zero entries, then set lower_median_dix to -1

  int lower_median_idx = upper_median_idx - 1;
  while (lower_median_idx >= 0 &&  
        ((lower_median_idx == 0 && cum_vals[0] == 0) ||
	    (cum_vals[lower_median_idx] == cum_vals[lower_median_idx-1]))) {
	lower_median_idx--;
  }
  //LOG(INFO) << "upper_idx:" << upper_median_idx;
  //LOG(INFO) << "lower_idx:" << lower_median_idx;

  // Slight paradigm shift.  The mass in bucket 0 is uniformly distributed 
  // between 0-1.  So median_idx == 0 ->  0 < median_pos < 1
  Dtype lower_cum_perc = (lower_median_idx >= 0) ? cum_vals[lower_median_idx] / total : 0;
  Dtype upper_cum_perc = cum_vals[upper_median_idx] / total;
  //LOG(INFO) << "upper_cum_perc:" << upper_cum_perc;
  //LOG(INFO) << "lower_cum_perc:" << lower_cum_perc;

  Dtype diff = upper_cum_perc - lower_cum_perc;
  Dtype slope = diff / (upper_median_idx - lower_median_idx);
  //LOG(INFO) << "slope:" << slope;

  // y = x * diff + lower
  // .5 = x * diff + lower
  // .5 - lower = x * diff
  // x = (.5 - lower) / diff
  Dtype fractional_x = (.5 - lower_cum_perc) / slope;
  //LOG(INFO) << "partial:" << fractional_x;

  lower_idx_out = lower_median_idx;
  upper_idx_out = upper_median_idx;
  slope_out = slope;

  return lower_median_idx + 1 + fractional_x;
}

template <typename Dtype>
Dtype compute_grad(int i, const vector<Dtype>& cum_vals, int lower_idx, int upper_idx, Dtype slope) {
  Dtype d_lcp_d_i, d_ucp_d_i;  // derivatives of {lower,upper}_cum_perc wrt an input value in position i
  Dtype total = cum_vals[cum_vals.size()-1];

  if (i > lower_idx && i < upper_idx) {
    // edge case where the median falls over an input of 0, so the
	// median placement is a bit arbitrary.
	// If this input were non-zero, then lower_idx would be different
	// so we have a discontinuity
	return 0.;
  }

  if (i <= lower_idx) {
    // positive value.  both total and cum_vals[lower_idx] increase
	// derivative of (x + a) / (x + b)
    d_lcp_d_i = (total - cum_vals[lower_idx]) / (total * total);
  } else {
    // negative value.  only total increases, so cum_vals[lower_idx]/ toal decreases
	// derivative of a / (x + b)
    d_lcp_d_i = -1 * cum_vals[lower_idx] / (total * total);
  }

  if (i <= upper_idx) {
    // positive value.  both total and cum_vals[upper_idx] increase
	// derivative of (x + a) / (x + b)
    d_ucp_d_i = (total - cum_vals[upper_idx]) / (total * total);
  } else {
    // negative value.  only total increases, so cum_vals[upper_idx]/ toal decreases
	// derivative of a / (x + b)
    d_ucp_d_i = -1 * cum_vals[upper_idx] / (total * total);
  }

  // changing the cumlative percs changes the slope between the two
  // integer positions around the median point
  Dtype d_slope_d_i = (d_ucp_d_i - d_lcp_d_i) / (upper_idx - lower_idx);


  // apply division rule of calculus to 
  // median = lower_idx + (0.5 - lcp) / slope
  Dtype num = slope * (-1 * d_lcp_d_i) - (.5 - cum_vals[lower_idx] / total) * d_slope_d_i;
  Dtype denum = slope * slope;
  //LOG(INFO) << d_lcp_d_i << " " <<  d_ucp_d_i << " " << d_slope_d_i << " " << num << " " << denum;
  return num / denum;
}

template <typename Dtype>
void MedianCenterOfMassLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  const bool normalize = this->layer_param().center_param().normalize();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
	  Dtype center_h, center_w, total_mass;
	  // find the median y-coordinate

	  // compute a vector of cumulative row sums
	  vector<Dtype> cum_row_sums;
	  Dtype cum_row_sum = 0;
	  for (int h = 0; h < height; h++) {
	    for (int w = 0; w < width; w++) {
	      const int idx = ((n * channels + c) * height + h) * width + w;
	      Dtype val = bottom_data[idx];  // val is the mass at (h,w)
		  cum_row_sum += val;
		}
		cum_row_sums.push_back(cum_row_sum);
	  }
	  total_mass = cum_row_sum;

      // compute vector of cumulative col sums
	  vector<Dtype> cum_col_sums;
	  Dtype cum_col_sum = 0;
	  for (int w = 0; w < width; w++) {
	    for (int h = 0; h < height; h++) {
	      const int idx = ((n * channels + c) * height + h) * width + w;
	      Dtype val = bottom_data[idx];  // val is the mass at (h,w)
		  cum_col_sum += val;
		}
		cum_col_sums.push_back(cum_col_sum);
	  }

	  if (total_mass) {
		// find the median position in cum_row_sums
		int h_lower_idx, h_upper_idx, w_lower_idx, w_upper_idx;
		Dtype h_slope, w_slope;
	    center_h = find_median_position(cum_row_sums, h_lower_idx, h_upper_idx, h_slope);
	    center_w = find_median_position(cum_col_sums, w_lower_idx, w_upper_idx, w_slope);
	  } else {
		// undefined, so choose an arbitrary location
	    center_h = height / 2;
	    center_w = width / 2;
	  }

	  if (normalize) {
	    center_h /= height;
	    center_w /= width;
	  }

	  top_data[(n * channels + c) * 2] = center_h;
	  top_data[(n * channels + c) * 2 + 1] = center_w;
	  //LOG(INFO) << "center_h, center_w, total_mass: " << center_h << ", " << center_w << ", " << total_mass;
    }
  }
}

template <typename Dtype>
void MedianCenterOfMassLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  memset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count());

  const Dtype* bottom_data = bottom[0]->cpu_data();

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const bool normalize = this->layer_param().center_param().normalize();

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
	  Dtype total_mass;

	  Dtype center_h_diff = top_diff[(n * channels + c) * 2];
	  Dtype center_w_diff = top_diff[(n * channels + c) * 2 + 1];
	  if (normalize) {
		center_h_diff /= height;
		center_w_diff /= width;
	  }
	  // find the median y-coordinate

	  // compute a vector of cumulative row sums
	  vector<Dtype> cum_row_sums;
	  Dtype cum_row_sum = 0;
	  for (int h = 0; h < height; h++) {
	    for (int w = 0; w < width; w++) {
	      const int idx = ((n * channels + c) * height + h) * width + w;
	      Dtype val = bottom_data[idx];  // val is the mass at (h,w)
		  cum_row_sum += val;
		}
		cum_row_sums.push_back(cum_row_sum);
	  }
	  total_mass = cum_row_sum;

      // compute vector of cumulative col sums
	  vector<Dtype> cum_col_sums;
	  Dtype cum_col_sum = 0;
	  for (int w = 0; w < width; w++) {
	    for (int h = 0; h < height; h++) {
	      const int idx = ((n * channels + c) * height + h) * width + w;
	      Dtype val = bottom_data[idx];  // val is the mass at (h,w)
		  cum_col_sum += val;
		}
		cum_col_sums.push_back(cum_col_sum);
	  }
	  
	  if (total_mass) {
		// find the median position in cum_row_sums
		int h_lower_idx, h_upper_idx, w_lower_idx, w_upper_idx;
		Dtype h_slope, w_slope;
	    find_median_position(cum_row_sums, h_lower_idx, h_upper_idx, h_slope);
	    find_median_position(cum_col_sums, w_lower_idx, w_upper_idx, w_slope);
		//LOG(INFO) << h_slope << "  " << w_slope;
	    for (int h = 0; h < height; h++) {
	      for (int w = 0; w < width; w++) {
		    const int idx = ((n * channels + c) * height + h) * width + w;
			Dtype d_center_h = compute_grad(h, cum_row_sums, h_lower_idx, h_upper_idx, h_slope);
			Dtype d_center_w = compute_grad(w, cum_col_sums, w_lower_idx, w_upper_idx, w_slope);
		    bottom_diff[idx] = center_h_diff * d_center_h + center_w_diff * d_center_w;
		  }
		}
	  } // end if(total_mass) 
	  // the else case would just set the derivative to 0, which is already done

    } // for channels 
  } // for num
} // Backward_cpu

INSTANTIATE_CLASS(MedianCenterOfMassLayer);
REGISTER_LAYER_CLASS(MedianCenterOfMass);

}  // namespace caffe
