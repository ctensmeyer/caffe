// Copyright 2017 Chris Tensmeyer

#include <iostream>  // NOLINT(readability/streams)
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PowCenterOfMassLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void PowCenterOfMassLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
double eval_residual(const double pos, const vector<Dtype> vals, const Dtype pow) {
  double residual = 0;

  for (int i = 0; i < vals.size(); i++) {
    double sign = std::copysign(1, i - pos);
	double mag = std::pow(std::abs(i - pos), pow);
	residual += sign * mag * vals[i];
  }
  return residual;
}

template <typename Dtype>
double find_center2(const vector<Dtype> vals, const Dtype pow, const double tol, double cur_estimate) {

  Dtype total = 0;
  if (cur_estimate < 0) {
    Dtype linear = 0;

    for (int i = 0; i < vals.size(); i++) {
      linear += i * vals[i];
      total += vals[i];
      //LOG(INFO) << i << ": " << vals[i];
    }
    // this is the CoM with pow=1
    cur_estimate = linear / total; 
  } else {
    for (int i = 0; i < vals.size(); i++) {
      total += vals[i];
	}
  }

  double residual = eval_residual(cur_estimate, vals, pow);
  if (residual == 0) {
    return cur_estimate;
  }

  double upper_bound, lower_bound, mid;
  upper_bound = lower_bound = cur_estimate;
  if (residual < 0) {
	do {
	  lower_bound -= 1;
	} while(eval_residual(lower_bound, vals, pow) < 0);
  } else {
	do {
	  upper_bound += 1;
	} while(eval_residual(upper_bound, vals, pow) > 0);
  }

  // binary search over the interval (lower_bound, upper_bound)
  int i = 0;
  while ( (upper_bound - lower_bound) > tol) {
    mid = (upper_bound + lower_bound) / 2;
    residual = eval_residual(mid, vals, pow);
	if (residual > 0) {
	  lower_bound = mid;
	} else {
	  upper_bound = mid;
	}
	i++;
	//LOG(INFO) << i << ": " << (upper_bound - lower_bound) << "\t" << residual << "\t" << mid;
  }

  return (upper_bound + lower_bound) / 2;
}

template <typename Dtype>
Dtype find_center(const vector<Dtype> vals, const Dtype pow, const Dtype tol, Dtype cur_estimate) {

  Dtype total = 0;
  if (cur_estimate < 0) {
    Dtype linear = 0;

    for (int i = 0; i < vals.size(); i++) {
      linear += i * vals[i];
      total += vals[i];
      //LOG(INFO) << i << ": " << vals[i];
    }
    // this is the CoM with pow=1
    cur_estimate = linear / total; 
  } else {
    for (int i = 0; i < vals.size(); i++) {
      total += vals[i];
	}
  }

  Dtype residual = eval_residual(cur_estimate, vals, pow);
  Dtype step = residual / total;
  Dtype prev_residual;
  Dtype lr = 1;
  int i = 0;
  while (lr * std::abs(step) >= tol) {
    cur_estimate += lr * step;
	i++;
    //LOG(INFO) << i << "\tcur_pos: " << cur_estimate << "\t" << "res: " << residual;

    prev_residual = residual;
    residual = eval_residual(cur_estimate, vals, pow);
	if (std::abs(residual) >= std::abs(prev_residual)) {
	  lr *= 0.5;
	} else {
	  lr *= 1.1;
	}
    // the next step
	step = residual / total;
  }
  //LOG(INFO) << i << "\tcur_pos: " << cur_estimate << "\t" << "res: " << residual;

  return cur_estimate;
}

template <typename Dtype>
void PowCenterOfMassLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  const bool normalize = this->layer_param().center_param().normalize();
  const Dtype pow = this->layer_param().center_param().pow();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  const double tol = 1e-8;

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
	  double center_h, center_w, total_mass;

	  // compute a vector of row sums
	  vector<Dtype> row_sums;
	  total_mass = 0;
	  for (int h = 0; h < height; h++) {
	    Dtype row_sum = 0;
	    for (int w = 0; w < width; w++) {
	      const int idx = ((n * channels + c) * height + h) * width + w;
	      Dtype val = bottom_data[idx];  // val is the mass at (h,w)
		  row_sum += val;
		  total_mass += val;
		}
		row_sums.push_back(row_sum);
	  }

      // compute vector of col sums
	  vector<Dtype> col_sums;
	  for (int w = 0; w < width; w++) {
	    Dtype col_sum = 0;
	    for (int h = 0; h < height; h++) {
	      const int idx = ((n * channels + c) * height + h) * width + w;
	      Dtype val = bottom_data[idx];  // val is the mass at (h,w)
		  col_sum += val;
		}
		col_sums.push_back(col_sum);
	  }

	  if (total_mass) {
	    center_h = find_center2(row_sums, pow, tol, (Dtype)-1.);
	    center_w = find_center2(col_sums, pow, tol, (Dtype)-1.);
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
void PowCenterOfMassLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
  const Dtype pow = this->layer_param().center_param().pow();
  const double tol = 1e-8;

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
	  double total_mass;

	  Dtype center_h_diff = top_diff[(n * channels + c) * 2];
	  Dtype center_w_diff = top_diff[(n * channels + c) * 2 + 1];
	  if (normalize) {
		center_h_diff /= height;
		center_w_diff /= width;
	  }

	  // compute a vector of row sums
	  vector<Dtype> row_sums;
	  total_mass = 0;
	  int non_zero = 0;;
	  for (int h = 0; h < height; h++) {
	    Dtype row_sum = 0;
	    for (int w = 0; w < width; w++) {
	      const int idx = ((n * channels + c) * height + h) * width + w;
	      Dtype val = bottom_data[idx];  // val is the mass at (h,w)
		  row_sum += val;
		  total_mass += val;
		  if (val > 0) {
		    non_zero++;
		  }
		}
		row_sums.push_back(row_sum);
	  }

      // compute vector of col sums
	  vector<Dtype> col_sums;
	  for (int w = 0; w < width; w++) {
	    Dtype col_sum = 0;
	    for (int h = 0; h < height; h++) {
	      const int idx = ((n * channels + c) * height + h) * width + w;
	      Dtype val = bottom_data[idx];  // val is the mass at (h,w)
		  col_sum += val;
		}
		col_sums.push_back(col_sum);
	  }

	  
	  if (total_mass) {
	  	double eps = total_mass / non_zero / 100;
	  	//double eps = 1e-2;
	    vector<double> col_dervs;
	    vector<double> row_dervs;

	    double center_h = find_center2(row_sums, pow, tol, (double)-1.);
	    double center_w = find_center2(col_sums, pow, tol, (double)-1.);
		LOG(INFO) << center_h << " " << center_w;

        // compute row sum derivatives w/ finite differencing
	    for (int h = 0; h < height; h++) {
		  Dtype original_val = row_sums[h];

		  row_sums[h] = original_val + eps;
	      double upper_center_h = find_center2(row_sums, pow, tol, (double)-1.);//center_h);

		  row_sums[h] = original_val - eps;
	      double lower_center_h = find_center2(row_sums, pow, tol, (double)-1.);//center_h);

		  double derv = (upper_center_h - lower_center_h) / (2*eps);
		  //LOG(INFO) << derv;
		  row_dervs.push_back(derv);
		}

        // compute col sum derivatives w/ finite differencing
	    for (int w = 0; w < width; w++) {
		  Dtype original_val = col_sums[w];

		  col_sums[w] = original_val + eps;
	      double upper_center_w = find_center2(col_sums, pow, tol, (double)-1.);//center_w);

		  col_sums[w] = original_val - eps;
	      double lower_center_w = find_center2(col_sums, pow, tol, (double)-1.);//center_w);
		  col_sums[w] = original_val;

		  double derv = (upper_center_w - lower_center_w) / (2*eps);
		  LOG(INFO) << upper_center_w << " " << eps;
		  LOG(INFO) << derv;
		  col_dervs.push_back(derv);
		}

	    for (int h = 0; h < height; h++) {
	      for (int w = 0; w < width; w++) {
		    const int idx = ((n * channels + c) * height + h) * width + w;
			double d_center_h = row_dervs[h];
			double d_center_w = col_dervs[w];
		    bottom_diff[idx] = center_h_diff * d_center_h + center_w_diff * d_center_w;
		  }
		}
	  } // end if(total_mass) 
	  // the else case would just set the derivative to 0, which is already done

    } // for channels 
  } // for num
} // Backward_cpu

INSTANTIATE_CLASS(PowCenterOfMassLayer);
REGISTER_LAYER_CLASS(PowCenterOfMass);

}  // namespace caffe
