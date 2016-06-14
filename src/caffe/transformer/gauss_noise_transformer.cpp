
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/random.hpp>

#include <string>
#include <vector>

#include "caffe/image_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {


template <typename Dtype>
void GaussNoiseImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  const int in_channels = in.channels();
  const int in_height = in.rows;
  const int in_width = in.cols;
  // out is same dims as in, but must be float
  out.create(in.size(), CV_32F | (0x18 & in.type()));

  vector<int> shape;
  shape.push_back(in_channels);
  shape.push_back(in_height);
  shape.push_back(in_width);
  rand_mask_->Reshape(shape);
  Dtype* rand_data = rand_mask_->mutable_cpu_data();
  this->RandGauss(in_channels * in_height * in_width, 0, cur_std_dev_, rand_data);


  // uses the opencv random state
  //cv::randn(out, in, cur_std_dev_);
  for (int h = 0; h < in_height; ++h) {
    // channel values are 1 byte wide (uchar)
	if (in.elemSize1() == 1) {
      const uchar* in_ptr = in.ptr<uchar>(h);
      float* out_ptr = out.ptr<float>(h);
      int index = 0;
      for (int w = 0; w < in_width; ++w) {
        for (int c = 0; c < in_channels; ++c) {
  	      out_ptr[index] = (in_ptr[index] + rand_data[h * in_height + index]);
          //DLOG(INFO) << "c: " << c << " h: " << h << " w: " << w << " index: " << index << " in_val: " << ((float)in_ptr[index]) << " + " << rand_data[index] << " = " << out_ptr[index];
  	      index++;
        }
      }
	}  else if (in.elemSize1() == 4) {
      const float* in_ptr = in.ptr<float>(h);
      float* out_ptr = out.ptr<float>(h);
      int index = 0;
      for (int w = 0; w < in_width; ++w) {
        for (int c = 0; c < in_channels; ++c) {
  	      out_ptr[index] = (in_ptr[index] + rand_data[h * in_height + index]);
          //DLOG(INFO) << "c: " << c << " h: " << h << " w: " << w << " index: " << index << " in_val: " << ((float)in_ptr[index]) << " + " << rand_data[index] << " = " << out_ptr[index];
  	      index++;
        }
      }
	}
  }
}

template <typename Dtype>
void GaussNoiseImageTransformer<Dtype>::SampleTransformParams(const vector<int>& in_shape) {
  ImageTransformer<Dtype>::SampleTransformParams(in_shape);

  CHECK_GT(param_.std_dev_size(), 0) << "Must specify std_dev";
  CHECK_LE(param_.std_dev_size(), 2) << "Cannot specify more than 2 values for std_dev";
  
  if (param_.std_dev_size() == 1) {
    cur_std_dev_ = param_.std_dev(0);
  } else {
    Dtype min_std, max_std;
    min_std = param_.std_dev(0);
    max_std = param_.std_dev(1);
	this->RandFloat(1, min_std, max_std, &cur_std_dev_);
  }

  PrintParams();
}

template <typename Dtype>
void GaussNoiseImageTransformer<Dtype>::PrintParams() {
  ImageTransformer<Dtype>::PrintParams();
  DLOG(INFO) << "PrintParams (" << this << ") " << "\tcur noise std_dev: " << cur_std_dev_;
}

INSTANTIATE_CLASS(GaussNoiseImageTransformer);

}  // namespace caffe
