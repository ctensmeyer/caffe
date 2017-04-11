#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
inline Dtype sigmoid(Dtype x, Dtype slope, Dtype center, Dtype offset, Dtype sign) {
  return 1. / (1. + exp(sign * slope * (x - (center + offset))));
}

template <typename Dtype>
inline Dtype sigmoid_d_slope(Dtype x, Dtype slope, Dtype center, Dtype offset, Dtype sign) {
  Dtype a = -1 * exp(sign * slope * (x - (center + offset))) * (sign * (x - (center + offset)));
  Dtype b = sigmoid(x, slope, center, offset, sign);
  b = b * b; // square it
  return a * b;
}

template <typename Dtype>
inline Dtype sigmoid_d_offset(Dtype x, Dtype slope, Dtype center, Dtype offset, Dtype sign) {
  Dtype a = exp(sign * slope * (x - (center + offset))) * (sign * slope);
  Dtype b = sigmoid(x, slope, center, offset, sign);
  b = b * b; // square it
  return a * b;
}


template <typename Dtype>
void RelativeDarknessLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int spatial_size = height * width;

  // assumes kernel_size_ is odd
  const int size = (kernel_size_ - 1) / 2;

  this->FixParams();
  const Dtype* params = this->blobs_[0]->cpu_data();
  const Dtype a_l = params[AL_IDX_];
  const Dtype a_m1 = params[AM1_IDX_];
  const Dtype a_m2 = params[AM2_IDX_];
  const Dtype a_u = params[AU_IDX_];
  const Dtype w_l = params[WL_IDX_];
  const Dtype w_r = params[WR_IDX_];

  caffe_set(top[0]->count(), (Dtype) 0, top_data);
  for (int n = 0; n < num; n++) {
    int bottom_num_offset = n * 1 * height * width;
	int top_num_offset = n * 3 * height * width;
	for (int h = 0; h < height; h++) {
	  for (int w = 0; w < width; w++) {
	    Dtype center_val = bottom_data[bottom_num_offset + h * height + w];
	    for (int i = -size; i <= size; i++) {
		  if (h + i < 0 || h + i >= height) {
		    continue;
		  }
		  for (int j = -size; j <= size; j++) {
		    if (w + j < 0 || w + j >= width || (j == 0 && i == 0)) {
			  // skip out of bounds and center pixel
		      continue;
		    }
	        Dtype cur_val = bottom_data[bottom_num_offset + (h + i) * height + (w + j)];
		    Dtype lower_val = sigmoid(cur_val, a_l, center_val, -1 * w_l, (Dtype) +1.0);
			Dtype middle_val_left = sigmoid(cur_val, a_m1, center_val, -1 * w_l, (Dtype) -1.0);
		    Dtype middle_val_right = sigmoid(cur_val, a_m2, center_val, w_r, (Dtype) +1.0);
		    Dtype upper_val = sigmoid(cur_val, a_u, center_val, w_r, (Dtype) -1.0);

			top_data[top_num_offset + LOWER_OFFSET_ * spatial_size + h * height + w] = lower_val;
			top_data[top_num_offset + UPPER_OFFSET_ * spatial_size + h * height + w] = upper_val;

			if (middle_val_left < middle_val_right) {
			  top_data[top_num_offset + MIDDLE_OFFSET_ * spatial_size + h * height + w] = middle_val_left;
			} else {
			  top_data[top_num_offset + MIDDLE_OFFSET_ * spatial_size + h * height + w] = middle_val_right;
			}
		  }
		}
	  }
	}
  }
}

/*
Designed to only backpropagate to the parameters, not the bottom blob
*/
template <typename Dtype>
void RelativeDarknessLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_data();
  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int spatial_size = height * width;

  // assumes kernel_size_ is odd
  const int size = (kernel_size_ - 1) / 2;

  const Dtype* params = this->blobs_[0]->cpu_data();
  const Dtype a_l = params[AL_IDX_];
  const Dtype a_m1 = params[AM1_IDX_];
  const Dtype a_m2 = params[AM2_IDX_];
  const Dtype a_u = params[AU_IDX_];
  const Dtype w_l = params[WL_IDX_];
  const Dtype w_r = params[WR_IDX_];
  Dtype* params_diff = this->blobs_[0]->mutable_cpu_diff();

  // Propagte to param
  if (this->param_propagate_down_[0]) {
    for (int n = 0; n < num; n++) {
      int bottom_num_offset = n * 1 * height * width;
      int top_num_offset = n * 3 * height * width;
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          Dtype center_val = bottom_data[bottom_num_offset + h * height + w];
		  Dtype lower_diff = top_diff[top_num_offset + LOWER_OFFSET_ * spatial_size + h * height + w];
		  Dtype middle_diff = top_diff[top_num_offset + MIDDLE_OFFSET_ * spatial_size + h * height + w];
		  Dtype upper_diff = top_diff[top_num_offset + UPPER_OFFSET_ * spatial_size + h * height + w];
          for (int i = -size; i <= size; i++) {
      	    if (h + i < 0 || h + i >= height) {
      	      continue;
      	    }
      	    for (int j = -size; j <= size; j++) {
      	      if (w + j < 0 || w + j >= width || (j == 0 && i == 0)) {
      	        // skip out of bounds and center pixel
      	        continue;
      	      }
	          Dtype cur_val = bottom_data[bottom_num_offset + (h + i) * height + (w + j)];

		      // lower 
		      params_diff[AL_IDX_] += lower_diff * sigmoid_d_slope(cur_val, a_l, center_val, -1 * w_l, (Dtype) +1.0);
		      params_diff[WL_IDX_] += lower_diff * sigmoid_d_offset(cur_val, a_l, center_val, -1 * w_l, (Dtype) +1.0);

		      // upper
		      params_diff[AU_IDX_] += upper_diff * sigmoid_d_slope(cur_val, a_u, center_val, w_r, (Dtype) -1.0);
		      params_diff[WL_IDX_] += upper_diff * sigmoid_d_offset(cur_val, a_u, center_val, w_r, (Dtype) -1.0);

		      // middle
		      Dtype middle_val_left = sigmoid(cur_val, a_m1, center_val, -1 * w_l, (Dtype) -1.0);
		      Dtype middle_val_right = sigmoid(cur_val, a_m2, center_val, w_r, (Dtype) +1.0);
		      if (middle_val_left < middle_val_right) {
		        params_diff[AM1_IDX_] += middle_diff * sigmoid_d_slope(cur_val, a_m1, center_val, -1 * w_l, (Dtype) -1.0);
		        params_diff[WL_IDX_] += middle_diff * sigmoid_d_offset(cur_val, a_m1, center_val, -1 * w_l, (Dtype) -1.0);
		      } else {
		        params_diff[AM2_IDX_] += middle_diff * sigmoid_d_slope(cur_val, a_m2, center_val, w_r, (Dtype) +1.0);
		        params_diff[WR_IDX_] += middle_diff * sigmoid_d_offset(cur_val, a_m2, center_val, w_r, (Dtype) +1.0);
		      }
      	    }
      	  }
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RelativeDarknessLayer);

}  // namespace caffe
