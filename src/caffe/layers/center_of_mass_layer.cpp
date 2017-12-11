// Copyright 2017 Chris Tensmeyer

#include <iostream>  // NOLINT(readability/streams)
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CenterOfMassLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_back_ = 0;
}

template <typename Dtype>
void CenterOfMassLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
void CenterOfMassLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* aux_data = aux_.mutable_cpu_data();

  // num_iters is the number of times to computer the Center of Mass
  // Each iteration, we restrict computation to only those points
  // within $radius of the CoM computed in the last iteration.
  // For the first iteration (i == 0), the all points in the grid
  // are considered.  Multiple iterations may be useful because
  // non-zero points far from the CoM have a larger influence that
  // those close to the CoM.
  Dtype num_iters = this->layer_param().center_param().num_iters();
  Dtype radius = this->layer_param().center_param().radius();
  Dtype min_multiple_iters = this->layer_param().center_param().min_multiple_iters();
  const Dtype radius_sq = radius * radius;
  // if we don't restrict the radius, the center dosen't change
  // with more iterations
  if (radius <= 0 || num_back_ < min_multiple_iters) {
	num_iters = 1;
  }

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
	  Dtype prev_center_h = 0;
	  Dtype prev_center_w = 0;
	  Dtype center_h, center_w, total_mass;
	  for (int i = 0; i < num_iters; i++) {
	    center_h = 0;
	    center_w = 0;
	    total_mass = 0;
        for (int h = 0; h < height; ++h) {
	      for (int w = 0; w < width; ++w) {
		    if (i > 0) {
			  // for every iteration except the first,
			  // we restrict the grid to those points within distance
			  // $radius to (prev_center_h, prev_center_w)
			  Dtype dist_sq = (h - prev_center_h) * (h - prev_center_h) +
			                  (w - prev_center_w) * (w - prev_center_w);
			  if (dist_sq > radius_sq) {
			    continue;
			  }
			}
	        const int idx = ((n * channels + c) * height + h) * width + w;
	        Dtype val = bottom_data[idx];  // val is the mass at (h,w)
	        center_h += h * val;
	        center_w += w * val;
	        total_mass += val;
	      }
        }
	    if (total_mass != 0) {
	      center_h /= total_mass;
	      center_w /= total_mass;
	    } else {
		  // undefined if the total mass is 0
		  // default to the center of the image, but there are likely more intelligent
		  // defaults that could be derived from the data
	      center_h = (height - 1) / 2.;
	      center_w = (width - 1) / 2.;
	    }
		// save values for next iterations
		prev_center_h = center_h;
		prev_center_w = center_w;
	  }

	  top_data[(n * channels + c) * 2] = center_h;
	  top_data[(n * channels + c) * 2 + 1] = center_w;
	  aux_data[(n * channels + c)] = total_mass;  // so it doesn't need to be recomputed in backwards step
	  //LOG(INFO) << "center_h, center_w, total_mass: " << center_h << ", " << center_w << ", " << total_mass;
    }
  }
}

template <typename Dtype>
void CenterOfMassLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  num_back_++;
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

}

INSTANTIATE_CLASS(CenterOfMassLayer);
REGISTER_LAYER_CLASS(CenterOfMass);

}  // namespace caffe
