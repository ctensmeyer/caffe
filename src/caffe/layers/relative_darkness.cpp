#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void RelativeDarknessLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4)
      << "Number of axes of bottom blob must be 4.";
  CHECK_EQ(bottom[0]->channels(), 1)
      << "Bottom blob must have only a single channel";
  RelativeDarknessParameter rd_param = this->layer_param().relative_darkness_param();
  kernel_size_ = rd_param.kernel_size();
  CHECK_GT(kernel_size_, 0)
      << "Kernel Size must be greater than 0";
  CHECK_EQ(kernel_size_ % 2, 1)
      << "Kernel Size must be an odd integer";

  min_param_value_ = rd_param.min_param_value();
  CHECK_GT(min_param_value_, 0)
      << "Min parameter value must be positive";

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
	vector<int> shape;
	shape.push_back(6);
	shape.push_back(1);
    this->blobs_[0].reset(new Blob<Dtype>(shape));
    shared_ptr<Filler<Dtype> > filler;
    if (rd_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(rd_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(0.25);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
  }
  //FixParams();
  LOG(INFO) << this->blobs_[0]->count();

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void RelativeDarknessLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4)
      << "Number of axes of bottom blob must be 4.";
  CHECK_EQ(bottom[0]->channels(), 1)
      << "Bottom blob must have only a single channel";

  vector<int> shape;
  shape.push_back(bottom[0]->num());
  shape.push_back(3);
  shape.push_back(bottom[0]->height());
  shape.push_back(bottom[0]->width());
  top[0]->Reshape(shape);

}

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
void RelativeDarknessLayer<Dtype>::FixParams() {
  Dtype* params = this->blobs_[0]->mutable_cpu_data();
  for (int i = 0; i < 6; i++) {
    if (params[i] < min_param_value_) {
	  params[i] =  min_param_value_;
	}
  }
}

template <typename Dtype>
void RelativeDarknessLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int spatial_size = height * width;

  // assumes kernel_size_ is odd
  const int size = (kernel_size_ - 1) / 2;
  const Dtype norm = 1. / kernel_size_ / kernel_size_;
  //LOG(INFO) << "Size: " << size;

  //this->FixParams();
  const Dtype* params = this->blobs_[0]->cpu_data();
  const Dtype a_l = params[AL_IDX_];
  const Dtype a_m1 = params[AM1_IDX_];
  const Dtype a_m2 = params[AM2_IDX_];
  const Dtype a_u = params[AU_IDX_];
  const Dtype w_l = params[WL_IDX_];
  const Dtype w_r = params[WR_IDX_];
  //LOG(INFO) << "Params: " << a_l << " " << a_m1 << " " << a_m2 << " " << a_u << " " << w_l << " " << w_r;
  

  caffe_set(top[0]->count(), (Dtype) 0, top_data);
  for (int n = 0; n < num; n++) {
    int bottom_num_offset = n * 1 * spatial_size;
	int top_num_offset = n * 3 * spatial_size;
	for (int h = 0; h < height; h++) {
	  for (int w = 0; w < width; w++) {
	    Dtype center_val = bottom_data[bottom_num_offset + h * height + w];
		//LOG(INFO) << "center_val (" << h << "," << w << "): " << center_val;
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
			/*
			LOG(INFO) << "\ti,j: " << i << ", " << j;
			LOG(INFO) << "\tcur_val: " << cur_val;
			LOG(INFO) << "\tlower_val: " << lower_val;
			LOG(INFO) << "\tmiddle_left_val: " << middle_val_left;
			LOG(INFO) << "\tmiddle_right_val: " << middle_val_right;
			LOG(INFO) << "\tupper_val: " << upper_val;
			*/

			top_data[top_num_offset + LOWER_OFFSET_ * spatial_size + h * height + w] += norm * lower_val;
			top_data[top_num_offset + UPPER_OFFSET_ * spatial_size + h * height + w] += norm * upper_val;

			if (middle_val_left < middle_val_right) {
			  top_data[top_num_offset + MIDDLE_OFFSET_ * spatial_size + h * height + w] += norm * middle_val_left;
			} else {
			  top_data[top_num_offset + MIDDLE_OFFSET_ * spatial_size + h * height + w] += norm * middle_val_right;
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
void RelativeDarknessLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int spatial_size = height * width;

  // assumes kernel_size_ is odd
  const int size = (kernel_size_ - 1) / 2;
  const Dtype norm = 1. / kernel_size_ / kernel_size_;
  //LOG(INFO) << "Size: " << size;

  const Dtype* params = this->blobs_[0]->cpu_data();
  const Dtype a_l = params[AL_IDX_];
  const Dtype a_m1 = params[AM1_IDX_];
  const Dtype a_m2 = params[AM2_IDX_];
  const Dtype a_u = params[AU_IDX_];
  const Dtype w_l = params[WL_IDX_];
  const Dtype w_r = params[WR_IDX_];
  Dtype* params_diff = this->blobs_[0]->mutable_cpu_diff();
  //LOG(INFO) << "Starting Backwards";

  // Propagte to param
  if (this->param_propagate_down_[0]) {
    for (int n = 0; n < num; n++) {
      int bottom_num_offset = n * 1 * spatial_size;
      int top_num_offset = n * 3 * spatial_size;
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          Dtype center_val = bottom_data[bottom_num_offset + h * height + w];
		  Dtype lower_diff = top_diff[top_num_offset + LOWER_OFFSET_ * spatial_size + h * height + w];
		  Dtype middle_diff = top_diff[top_num_offset + MIDDLE_OFFSET_ * spatial_size + h * height + w];
		  Dtype upper_diff = top_diff[top_num_offset + UPPER_OFFSET_ * spatial_size + h * height + w];
		  //LOG(INFO) << lower_diff << " " << middle_diff << " " << upper_diff;
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
		      params_diff[AL_IDX_] += norm * lower_diff * sigmoid_d_slope(cur_val, a_l, center_val, -1 * w_l, (Dtype) +1.0);
		      params_diff[WL_IDX_] -= norm * lower_diff * sigmoid_d_offset(cur_val, a_l, center_val, -1 * w_l, (Dtype) +1.0);

		      // upper
		      params_diff[AU_IDX_] += norm * upper_diff * sigmoid_d_slope(cur_val, a_u, center_val, w_r, (Dtype) -1.0);
		      params_diff[WR_IDX_] += norm * upper_diff * sigmoid_d_offset(cur_val, a_u, center_val, w_r, (Dtype) -1.0);

		      // middle
		      Dtype middle_val_left = sigmoid(cur_val, a_m1, center_val, -1 * w_l, (Dtype) -1.0);
		      Dtype middle_val_right = sigmoid(cur_val, a_m2, center_val, w_r, (Dtype) +1.0);
		      if (middle_val_left < middle_val_right) {
		        params_diff[AM1_IDX_] += norm * middle_diff * sigmoid_d_slope(cur_val, a_m1, center_val, -1 * w_l, (Dtype) -1.0);
		        params_diff[WL_IDX_] -= norm * middle_diff * sigmoid_d_offset(cur_val, a_m1, center_val, -1 * w_l, (Dtype) -1.0);
		      } else {
		        params_diff[AM2_IDX_] += norm * middle_diff * sigmoid_d_slope(cur_val, a_m2, center_val, w_r, (Dtype) +1.0);
		        params_diff[WR_IDX_] += norm * middle_diff * sigmoid_d_offset(cur_val, a_m2, center_val, w_r, (Dtype) +1.0);
		      }
      	    }
      	  }
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(RelativeDarknessLayer);
#endif

INSTANTIATE_CLASS(RelativeDarknessLayer);
REGISTER_LAYER_CLASS(RelativeDarkness);

}  // namespace caffe
