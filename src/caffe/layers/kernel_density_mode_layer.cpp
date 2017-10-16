// Copyright 2017 Chris Tensmeyer

#include <iostream>  // NOLINT(readability/streams)
#include <vector>
#include <cmath>
#include <limits>

#include "caffe/layer.hpp"
#include "caffe/layers/kernel_density_mode_layer.hpp"

namespace caffe {

double linear_kernel(double distance, double radius) {
  return (radius - distance) / radius;
}

double epanechnikov_kernel(double distance, double radius) {
  return (radius * radius - distance * distance) / (radius * radius); 
}

double d_linear_kernel(double radius) {
  return -1. / radius;
}

double d_epanechnikov_kernel(double distance, double radius) {
  return -2. * distance / (radius * radius); 
}

void get_range(double h, double w, double radius, 
			   int height, int width, int& min_h, int& max_h, 
			   int& min_w, int& max_w) {
  min_h = std::max((int) std::ceil(h - radius), 0);
  max_h = std::min((int) std::floor(h + radius), height - 1);
  min_w = std::max((int) std::ceil(w - radius), 0);
  max_w = std::min((int) std::floor(w + radius), width - 1);
}

template <typename Dtype>
double KernelDensityModeLayer<Dtype>::KernelVal(double distance) {
  switch(this->layer_param().kernel_param().kernel_type()) {
    case KernelDensityModeParameter_KernelType_LINEAR:
	{
	  return linear_kernel(distance, radius_);
	}
    case KernelDensityModeParameter_KernelType_EPANECHNIKOV:
	{
	  return epanechnikov_kernel(distance, radius_);
	}
	default:
	 CHECK_EQ(0, 1) << "Invalid Kernel Type"; 
	 return 0;
  }
  return 0;
}

template <typename Dtype>
double KernelDensityModeLayer<Dtype>::dKernelVal(double distance) {
  switch(this->layer_param().kernel_param().kernel_type()) {
    case KernelDensityModeParameter_KernelType_LINEAR:
	{
	  return d_linear_kernel(radius_);
	}
    case KernelDensityModeParameter_KernelType_EPANECHNIKOV:
	{
	  return d_epanechnikov_kernel(distance, radius_);
	}
	default:
	 CHECK_EQ(0, 1) << "Invalid Kernel Type"; 
	 return 0;
  }
  return 0;
}

template <typename Dtype>
void KernelDensityModeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  grad_thresh_ = this->layer_param().kernel_param().grad_thresh();
  radius_ = this->layer_param().kernel_param().radius();
  finite_diff_ = true;
  //finite_diff_ = false;

  int num_output = bottom[0]->shape(1);
  int kernel_size = (int) std::ceil(radius_);
  if (kernel_size % 2 == 0) {
    kernel_size++;
  }
  int pad_size = (kernel_size - 1) / 2;
  LayerParameter param;
  param.mutable_convolution_param()->set_num_output(num_output);
  param.mutable_convolution_param()->set_group(num_output);
  param.mutable_convolution_param()->add_kernel_size(kernel_size);
  param.mutable_convolution_param()->add_pad(pad_size);
  param.mutable_convolution_param()->set_bias_term(false);
  param.mutable_convolution_param()->add_stride(1);

  conv_layer_ = new ConvolutionLayer<Dtype>(param);
  conv_layer_->LayerSetUp(bottom, top);

  conv_bottom_vec_.push_back(bottom[0]);
  conv_top_vec_.push_back(&conv_top_);

  // Set the conv weights
  double center = kernel_size / 2.;
  Dtype* weights = conv_layer_->blobs()[0]->mutable_cpu_data();
  for (int n = 0; n < num_output; n++) {
    for (int h = 0; h < kernel_size; h++) {
      for (int w = 0; w < kernel_size; w++) {
	    double dist = std::sqrt( (h - center) * (h - center) + (w - center) * (w - center) );
		weights[((n * kernel_size) + h) * kernel_size + w] = this->KernelVal(dist);
      }
    }
  }
}



template <typename Dtype>
void KernelDensityModeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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

  conv_layer_->Reshape(conv_bottom_vec_, conv_top_vec_);
  for (int axis = 0; axis < 3; axis++) {
    CHECK_EQ(conv_bottom_vec_[0]->shape(axis), conv_top_vec_[0]->shape(axis)) <<
		"The output of the Conv Layer inside KernelDensityModeLayer must have the same " <<
		"shape as the input.  " << conv_bottom_vec_[0]->shape_string() << " vs " <<
		conv_top_vec_[0]->shape_string();
  }
  if (top.size() > 1) {
    top[1]->ReshapeLike(*(conv_top_vec_[0]));
  }
}


template <typename Dtype>
double KernelDensityModeLayer<Dtype>::LineSearch(const Dtype* grid, const int height, 
      const int width, double h, double w, double dh, double dw, int num_steps, 
	  double max_step, bool backwards) {

  // initialize three points
  double lower_step = 0;
  if (backwards) {
    lower_step = -1 * max_step;
  }
  double upper_step = max_step;
  double middle_step = (2 * lower_step + upper_step) / 3.;
  double lower_step_val = this->GridVal(grid, height, width, 
  	h + lower_step * dh, w + lower_step * dw);
  double upper_step_val = this->GridVal(grid, height, width, 
  	h + upper_step * dh, w + upper_step * dw);
  double middle_step_val = this->GridVal(grid, height, width, 
  	h + middle_step * dh, w + middle_step * dw);

  double step_size = 0;
  int original_num_steps = num_steps;

  for (int k = 0; k < num_steps; k++) {
    double middle_lower_step;
    double middle_upper_step;
    double middle_lower_step_val;
    double middle_upper_step_val;
  
    // stick another point halfway inbetween whichever is farther:
	// upper -> middle or lower -> middle
    if ( (upper_step - middle_step) > (middle_step - lower_step) ) {
      middle_upper_step = middle_step + (upper_step - middle_step) / 2.;
  	  middle_upper_step_val = this->GridVal(grid, height, width, 
  	    h + middle_upper_step * dh, w + middle_upper_step * dw);
  
  	  middle_lower_step = middle_step;
  	  middle_lower_step_val = middle_step_val;
    } else {
      middle_lower_step = lower_step + (middle_step - lower_step) / 2.;
  	  middle_lower_step_val = this->GridVal(grid, height, width, 
  	    h + middle_lower_step * dh, w + middle_lower_step * dw);
  
  	  middle_upper_step = middle_step;
  	  middle_upper_step_val = middle_step_val;
    }
	/*
	LOG(INFO) << "     ls " << k << " (" << lower_step << "," << lower_step_val << ") (" <<
			middle_lower_step << "," << middle_lower_step_val << ") (" << middle_upper_step << ","
			<< middle_upper_step_val << ") (" << upper_step << "," << upper_step_val << ")";
	*/
  
    // toss out an end point 
    if (middle_upper_step_val > middle_lower_step_val) {
      // bye bye lower_step
  	  // middle_lower -> lower
  	  // middle_upper -> middle
  	  lower_step = middle_lower_step;
  	  lower_step_val = middle_lower_step_val;
  
  	  middle_step = middle_upper_step;
  	  middle_step_val = middle_upper_step_val;
    } else {
      // bye bye upper_step
  	  // middle_upper -> upper
  	  // middle_lower -> middle
  	  upper_step = middle_upper_step;
  	  upper_step_val = middle_upper_step_val;
  
  	  middle_step = middle_lower_step;
  	  middle_step_val = middle_lower_step_val;
    }
	if (k == num_steps - 1) {
	  // last iter, maybe

      // return the step that gets us to the highest value
      if (upper_step_val > middle_step_val && upper_step_val > lower_step_val) {
        step_size = upper_step;
      } else if (middle_step_val > upper_step_val && middle_step_val > lower_step_val) {
        step_size = middle_step;
      } else {
        step_size = lower_step;
      }

	  if (step_size == 0 && num_steps == original_num_steps) {
	    num_steps += 10;
	  }
	}
  } 
  return step_size;
}

template <typename Dtype>
void KernelDensityModeLayer<Dtype>::GradientAscent(const Dtype* grid, const int height, 
      const int width, double initial_h, double initial_w, double& center_h, double& center_w,
	  double initial_step_size, int initial_line_search_iters, int max_iters) {
  //LOG(INFO) << "grad_thresh: " << grad_thresh_;
  //LOG(INFO) << "initial: " << initial_h << "," << initial_w;

  double cur_h = initial_h;
  double cur_w = initial_w;
  double dh, dw, grad_mag;
  int k = 0;
  double step_size = initial_step_size / 10.;
  do {
	//double val = this->GridVal(grid, height, width, cur_h, cur_w);
	this->dGridVal(grid, height, width, cur_h, cur_w, dh, dw);
	
	grad_mag = std::sqrt(dh * dh + dw * dw);
	double max_step = 1. / std::min<double>(1., std::sqrt(grad_mag));
	//if (max_step < grad_thresh_) max_step = grad_thresh_; 
	//if (max_step > 1.) max_step = 1.;
	
	int num_line_search_iters = initial_line_search_iters + std::min(k, 16) / 2;
	step_size = this->LineSearch(grid, height, width, cur_h, cur_w, dh, dw,
	                                   num_line_search_iters, max_step, false);
	/*
    LOG(INFO) << "iter: " << k << " grad_mag: " << grad_mag << " dh: " << dh << " dw: " << dw << " val: " << val;
    LOG(INFO) << "         cur_h: " << cur_h << " cur_w: " << cur_w << " step_size: " << step_size << " max_step: " << max_step;
	LOG(INFO) << "\n";
	*/
    cur_h += dh * step_size;
    cur_w += dw * step_size;
	k++;
  } while (grad_mag >= grad_thresh_ && k <= max_iters && step_size > 0);
  
  center_h = cur_h;
  center_w = cur_w;
}

template <typename Dtype>
double KernelDensityModeLayer<Dtype>::GridVal(const Dtype* grid, const int height, 
    const int width, double h, double w) {

  int min_h = 0, max_h = 0, min_w = 0, max_w = 0;
  get_range(h, w, radius_, height, width, min_h, max_h, min_w, max_w);

  double val = 0;
  for (int _h = min_h; _h <= max_h; _h++) {
    for (int _w = min_w; _w <= max_w; _w++) {
	  int idx = _h * height + _w;
	  double dist = std::sqrt( (h - _h) * (h - _h) + (w - _w) * (w - _w));
	  if (dist > radius_) {
	    continue;
	  }
	  double kernel_val = this->KernelVal(dist);
	  val += grid[idx] * kernel_val;
	}
  }
  return val;
}

template <typename Dtype>
void KernelDensityModeLayer<Dtype>::dGridVal(const Dtype* grid, const int height, 
    const int width, double h, double w, double& dh, double& dw) {

  int min_h = 0, max_h = 0, min_w = 0, max_w = 0;
  get_range(h, w, radius_, height, width, min_h, max_h, min_w, max_w);

  dh = 0;
  dw = 0;
  for (int _h = min_h; _h <= max_h; _h++) {
    for (int _w = min_w; _w <= max_w; _w++) {
	  int idx = _h * height + _w;
	  double dist = std::sqrt( (h - _h) * (h - _h) + (w - _w) * (w - _w));
	  if (dist > radius_ || dist == 0) {
	    continue;
	  }

	  // the direction of the derivative vector is (_h - h, _w, w)
	  double d_dist_mag = this->dKernelVal(dist);

	  // project magnitude along axis aligned vectors
	  dh += grid[idx] * (h - _h) * d_dist_mag / dist;
	  dw += grid[idx] * (w - _w) * d_dist_mag / dist;
	}
  }
}

template <typename Dtype>
double KernelDensityModeLayer<Dtype>::SumMagdGridVal(const Dtype* grid, const int height, 
    const int width, double h, double w) {

  int min_h = 0, max_h = 0, min_w = 0, max_w = 0;
  get_range(h, w, radius_, height, width, min_h, max_h, min_w, max_w);

  double mag = 0;
  for (int _h = min_h; _h <= max_h; _h++) {
    for (int _w = min_w; _w <= max_w; _w++) {
	  int idx = _h * height + _w;
	  double dist = std::sqrt( (h - _h) * (h - _h) + (w - _w) * (w - _w));
	  if (dist > radius_ || dist == 0) {
	    continue;
	  }

	  // the direction of the derivative vector is (_h - h, _w, w)
	  double d_dist_mag = std::abs(this->dKernelVal(dist));
	  mag += grid[idx] * d_dist_mag;
	}
  }
  return mag;
}

template <typename Dtype>
void KernelDensityModeLayer<Dtype>::dOutdGrid(double h_star, double w_star, int h, 
      int w, double& dh_star_dg, double& dw_star_dg) {

  double dist = std::sqrt((h - h_star) * (h - h_star) + (w - w_star) * (w - w_star));
  if (dist <= radius_ && dist > 0) {
    double d_dist_mag = std::abs(this->dKernelVal(dist));
    dh_star_dg = (h - h_star) * d_dist_mag / dist;
    dw_star_dg = (w - w_star) * d_dist_mag / dist;
  } else {
    dh_star_dg = 0;
	dw_star_dg = 0;
  }
}

template <typename Dtype>
void KernelDensityModeLayer<Dtype>::FiniteDiff(Dtype* grid, int height, int width, double eps,
      double h_star, double w_star, int h, int w, double& dh_star_dg, double& dw_star_dg) {

  const int idx = h * height + w;
  const Dtype orig_input = grid[idx];

/*
  double h_star_upper = 0;
  double w_star_upper = 0;
      
  grid[idx] = orig_input + eps;
  // start gradient ascent at the previous maximum (h*, w*)
  this->GradientAscent(grid, height, width, h_star, w_star,
                       h_star_upper, w_star_upper, 2., 30, 100);
  double dh_star_dg_ga = (h_star_upper - h_star) / eps;
  double dw_star_dg_ga = (w_star_upper - w_star) / eps;
*/

  double step_size = 0.;
  int k = 0;
  while (step_size == 0 && k < 3) {
    k++;
    //LOG(INFO) << eps;

    // find derivative through finite differences
    grid[idx] = orig_input + eps;

    // unit vector from (h_star, w_star) to (h,w)
    double dh = (h - h_star);
    double dw = (w - w_star);
    double dist = std::sqrt(dh * dh + dw * dw);
    if (dist == 0) {
      dh_star_dg = 0;
      dw_star_dg = 0;
      return;
    }
      
    dh /= dist;
    dw /= dist;

    step_size = this->LineSearch(grid, height, width, h_star, w_star, 
    						dh, dw, 35, 2., false);
	/*						
	if (step_size == 0) {
	  LOG(INFO) << "Step Size 0";
	}
	*/

    dh_star_dg = step_size * dh / eps;
    dw_star_dg = step_size * dw / eps;

	eps *= 2;
  }

  // set the input back
  grid[idx] = orig_input;

/*
  double dh_diff = (dh_star_dg_ga - dh_star_dg) / std::abs( (dh_star_dg_ga + dh_star_dg) / 2);
  double dw_diff = (dw_star_dg_ga - dw_star_dg) / std::abs( (dw_star_dg_ga + dw_star_dg) / 2);
  LOG(INFO) << "diff_perc: (" << dh_diff << "," << dw_diff << ")" << " ls: (" << dh_star_dg << "," << dw_star_dg << ")";
*/
}

template <typename Dtype>
void KernelDensityModeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  conv_layer_->Forward(conv_bottom_vec_, conv_top_vec_);
  const Dtype* conv_top_data = conv_top_.cpu_data();

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
	  const int grid_offset = (n * channels + c) * height * width;
	  double initial_h = 0;
	  double initial_w = 0;
	  Dtype max = -1 * std::numeric_limits<Dtype>::max();
	  // find starting point
      for (int h = 0; h < height; ++h) {
	    for (int w = 0; w < width; ++w) {
		  const int idx = grid_offset + (h * width) + w;
		  if (conv_top_data[idx] > max) {
		    max = conv_top_data[idx];
			initial_h = h;
			initial_w = w;
		  }
		}
	  }
	  //LOG(INFO) << "Initial h,w: (" << initial_h << "," << initial_w << ")";
	  // gradient descent starting at (initial_h, initial_w)
	  double center_h, center_w;
	  this->GradientAscent(bottom_data + grid_offset, height, width, initial_h, initial_w,
	                         center_h, center_w, 1., 15, 100);

	  top_data[(n * channels + c) * 2] = center_h;
	  top_data[(n * channels + c) * 2 + 1] = center_w;
	  //LOG(INFO) << "n = " << n << " c = " << c << " grid_offset = " << grid_offset;
	  //LOG(INFO) << "initial_h, initial_w, max: " << initial_h << ", " << initial_w << ", " << max;
	  //LOG(INFO) << "center_h, center_w: " << center_h << ", " << center_w;
    }
  }
  if (top.size() > 1) {
    top[1]->CopyFrom(*(conv_top_vec_[0]));
  }
}


template <typename Dtype>
void KernelDensityModeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  memset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count());

  const Dtype* top_data = top[0]->cpu_data();

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
	  const Dtype dL_dh_star = top_diff[(n * channels + c) * 2];
	  const Dtype dL_dw_star = top_diff[(n * channels + c) * 2 + 1];
	  Dtype h_star = top_data[(n * channels + c) * 2];
	  Dtype w_star = top_data[(n * channels + c) * 2 + 1];

      /*
	  if (h_star == (int) h_star && w_star == (int) w_star) {
	    LOG(INFO) << "Tweak";
        h_star += 0.001;
        w_star += 0.001;
	  }
	  LOG(INFO) << "h*,w* = (" << h_star << "," << w_star << ")";
	  */

	  const int grid_offset = (n * channels + c) * height * width;
	  int min_h = 0, max_h = 0, min_w = 0, max_w = 0;
	  get_range(h_star, w_star, radius_, height, width,
	            min_h, max_h, min_w, max_w);

	  double total_val = 0;
	  int count = 0;
      for (int h = min_h; h <= max_h; ++h) {
	    for (int w = min_w; w <= max_w; ++w) {
          double dist_sq = (h - h_star) * (h - h_star) + (w - w_star) * (w - w_star);
          if (dist_sq > (radius_ * radius_)) {
		    continue;
		  }
		  const int idx = grid_offset + h * height + w;
		  total_val += bottom_data[idx];
		  count++;
		}
	  }
	  double avg_val = total_val / count;
	  //double eps = radius_ * avg_val / 2;
	  //double sum_mag = SumMagdGridVal(bottom_data + grid_offset, height, width, h_star, w_star);
	  double max_val = GridVal(bottom_data + grid_offset, height, width, h_star, w_star);
	  //LOG(INFO) << "    max_val = " << max_val;

	  double dh = 0, dw = 0;
	  dGridVal(bottom_data + grid_offset, height, width, h_star, w_star, dh, dw);
	  //double _grad_mag = std::sqrt( dh * dh + dw * dw );
	  //LOG(INFO) << "    dh,dw = (" << dh << "," << dw << "), mag = " << _grad_mag;

      for (int h = min_h; h <= max_h; ++h) {
	    for (int w = min_w; w <= max_w; ++w) {
          double dist_sq = (h - h_star) * (h - h_star) + (w - w_star) * (w - w_star);
          if (dist_sq > (radius_ * radius_)) {
		    continue;
		  }

		  const int idx = grid_offset + h * height + w;
		  double dh_star_dg = 0, dw_star_dg = 0;

		  if (finite_diff_) {
	        double eps = radius_ * avg_val / 2;
		    FiniteDiff(bottom_data + grid_offset, height, width, eps, h_star, w_star, 
		             h, w, dh_star_dg, dw_star_dg);
		  } else {
		    double dist = std::sqrt(dist_sq);
		    double eps = std::min<double>(0.01, dist);

			// if (h*,w*) were to move by eps towards (h,w)
			// how much increase in g(h,w) would it take for
			// v(h*,w*) == v( (h*,w*) + eps)
			double delta_h_star = h_star + eps * (h - h_star);
			double delta_w_star = w_star + eps * (w - w_star);
			double eps_max_val = GridVal(bottom_data + grid_offset, height, width, 
									 delta_h_star, delta_w_star);
			double delta_max_val = max_val - eps_max_val;
			double delta_dist = std::sqrt( (h - delta_h_star) * (h - delta_h_star) +
										   (w - delta_w_star) * (w - delta_w_star) );
			CHECK_LE(delta_dist, dist);
			double delta_kernel_val = KernelVal(delta_dist) - KernelVal(dist);
			double grid_delta = delta_max_val / delta_kernel_val;
			double grad_mag = eps / grid_delta;

			dh_star_dg = grad_mag * (h - h_star) / dist;
			dw_star_dg = grad_mag * (w - w_star) / dist;
		    
		    //this->dOutdGrid(h_star, w_star, h, w, dh_star_dg, dw_star_dg);
			//dh_star_dg *= KernelVal(radius_ / 2) / sum_mag;
			//dw_star_dg *= KernelVal(radius_ / 2) / sum_mag;
		  }
		  bottom_diff[idx] = dL_dh_star * dh_star_dg + dL_dw_star * dw_star_dg;
		  //bottom_diff[idx] = dh_star_dg;
		}
      }
    }
  }
}


INSTANTIATE_CLASS(KernelDensityModeLayer);
REGISTER_LAYER_CLASS(KernelDensityMode);

}  // namespace caffe
