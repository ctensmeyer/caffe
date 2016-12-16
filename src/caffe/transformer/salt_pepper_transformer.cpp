
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
void SaltPepperImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  const int in_channels = in.channels();
  const int in_height = in.rows;
  const int in_width = in.cols;
  // out is same dims and type as in
  out.create(in.size(), in.type());

  vector<int> shape;
  shape.push_back(in_channels);
  shape.push_back(in_height);
  shape.push_back(in_width);

  // sample values for each pixel
  rand_mask_->Reshape(shape);
  Dtype* rand_data = rand_mask_->mutable_cpu_data();
  this->RandFloat(in_channels * in_height * in_width, 0, 1, rand_data);

  // sample amount of salt/pepper noise
  Dtype percent_pixels;
  Dtype percent_salt;
  Dtype outcome;
  this->RandFloat(1, param_.min_percent_pixels(), param_.max_percent_pixels(), &percent_pixels);
  this->RandFloat(1, param_.min_percent_salt(), param_.max_percent_salt(), &percent_salt);

  for (int h = 0; h < in_height; ++h) {
    // channel values are 1 byte wide (uchar)
	if (in.elemSize1() == 1) {
      const uchar* in_ptr = in.ptr<uchar>(h);
      uchar* out_ptr = out.ptr<uchar>(h);
      int index = 0;
      for (int w = 0; w < in_width; ++w) {
        for (int c = 0; c < in_channels; ++c) {
		  if (rand_data[h * in_height + index] <= percent_pixels) {
			  this->RandFloat(1, 0, 1, &outcome);
			  out_ptr[index] = (outcome <= percent_salt) ? 255 : 0;
		  } else {
			  out_ptr[index] = in_ptr[index];
		  }
  	      index++;
        }
      }
	}  else if (in.elemSize1() == 4) {
      const float* in_ptr = in.ptr<float>(h);
      float* out_ptr = out.ptr<float>(h);
      int index = 0;
      for (int w = 0; w < in_width; ++w) {
        for (int c = 0; c < in_channels; ++c) {
		  if (rand_data[h * in_height + index] <= percent_pixels) {
			  this->RandFloat(1, 0, 1, &outcome);
			  out_ptr[index] = (outcome <= percent_salt) ? 255. : 0.;
		  } else {
			  out_ptr[index] = in_ptr[index];
		  }
  	      index++;
        }
      }
	}
  }
}


INSTANTIATE_CLASS(SaltPepperImageTransformer);

}  // namespace caffe
