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

  vector<int> shape2;
  shape2.push_back(num);
  shape2.push_back(channels);
  shape2.push_back(1);
  aux_.Reshape(shape2);
}

template <typename Dtype>
Dtype find_median_position(const vector<Dtype> vals) {
  int median_idx = 0;
  Dtype total = vals[vals.size()-1];
  for (int i = 0; i < vals.size(); i++) {
	Dtype val = vals[i];
	Dtype cum_perc = val / total;
	if (cum_perc > 0.5) {
	  median_idx = i;
	  break;
	}
  }
  // median_idx is now pointing to the first val with cumulative sum
  // greater than 50%, the median position is somewhere between
  // median_idx and median_idx-1.

  // Slight paradigm shift.  The mass in bucket 0 is uniformly distributed 
  // between 0-1.  So median_idx == 0 ->  0 < median_pos < 1
  Dtype lower_cum_perc;
  Dtype upper_cum_perc = vals[median_idx] / total;
  if (median_idx == 0) {
	lower_cum_perc = 0;
  } else {
	upper_cum_perc = vals[median_idx-1] / total;
  }
  Dtype diff = upper_cum_perc - lower_cum_perc;

  // y = x * diff + lower
  // .5 = x * diff + lower
  // .5 - lower = x * diff
  // x = (.5 - lower) / diff
  Dtype fractional_x = (.5 - lower_cum_perc) / diff;

  return median_idx + fractional_x;
}

template <typename Dtype>
void MedianCenterOfMassLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* aux_data = aux_.mutable_cpu_data();

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
	    center_h = find_median_position(cum_row_sums);
	    center_w = find_median_position(cum_col_sums);
	  } else {
	    center_h = height / 2;
	    center_w = width / 2;
	  }

	  top_data[(n * channels + c) * 2] = center_h;
	  top_data[(n * channels + c) * 2 + 1] = center_w;
	  aux_data[(n * channels + c)] = total_mass;  // so it doesn't need to be recomputed in backwards step
	  //LOG(INFO) << "center_h, center_w, total_mass: " << center_h << ", " << center_w << ", " << total_mass;
    }
  }
}

template <typename Dtype>
void MedianCenterOfMassLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
/*
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  memset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count());

  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* aux_data = aux_.cpu_data();

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  Dtype radius = this->layer_param().center_param().radius();
  Dtype min_multiple_iters = this->layer_param().center_param().min_multiple_iters();
  if (radius <= 0) {
    // make it so big, it won't matter
    radius = height + width + 1;
  }
  const Dtype radius_squared = radius * radius;
  const Dtype far_grad = this->layer_param().center_param().grad();

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
	  const Dtype center_h_diff = top_diff[(n * channels + c) * 2];
	  const Dtype center_w_diff = top_diff[(n * channels + c) * 2 + 1];
	  const Dtype center_h = top_data[(n * channels + c) * 2];
	  const Dtype center_w = top_data[(n * channels + c) * 2 + 1];
	  const Dtype total_mass = aux_data[n * channels + c];
      for (int h = 0; h < height; ++h) {
	    for (int w = 0; w < width; ++w) {
		  const int idx = ((n * channels + c) * height + h) * width + w;
		  Dtype dist_squared = (h - center_h) * (h - center_h) + 
		                       (w - center_w) * (w - center_w);
		  if (num_back_ < min_multiple_iters || dist_squared <= radius_squared) {
		    Dtype h_num = center_h * total_mass;
		    Dtype d_h_num = h;
		    Dtype w_num = center_w * total_mass;
		    Dtype d_w_num = w;
		    Dtype denum = total_mass;
		    Dtype d_denum = 1.;
		    Dtype d_center_h = (denum * d_h_num - h_num * d_denum) / (denum * denum);
		    Dtype d_center_w = (denum * d_w_num - w_num * d_denum) / (denum * denum);
		    bottom_diff[idx] = center_h_diff * d_center_h + center_w_diff * d_center_w;
		  } else {
		    // (h,w) is too far away from the center of mass
		    bottom_diff[idx] = far_grad;
		  }
		}
      }
    }
  }
*/

}

INSTANTIATE_CLASS(MedianCenterOfMassLayer);
REGISTER_LAYER_CLASS(MedianCenterOfMass);

}  // namespace caffe
