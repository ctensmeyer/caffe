
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
void LinearImageTransformer<Dtype>::LoadShiftFile() {
  // load in the shift image from disk

}

template <typename Dtype>
void LinearImageTransformer<Dtype>::ResizeShiftImage(const vector<int>& in_shape) {
  // reshape the shift image to match the current input size
  CHECK_GE(in_shape.size(), 2);
  CHECK_LE(in_shape.size(), 4);
  /*
  const int cur_height = shift_image_current_.rows;
  const int cur_width = shift_image_current_.height;
  const int cur_channels = shift_image_current.channels();

  int in_height = in_shape[in_shape.size() - 2];
  int in_width = in_shape[in_shape.size() - 1];

  if (param_.error_on_size_mismatch()) {
    CHECK_EQ(in_width, cur_width) << "Shift Image File width does not match input size";
	CHECK_EQ(in_height, cur_height) << "Shift Image File height does not match input size";
  }
  */

}

// assume out is the proper size...
// assume out is CV_32F
template <typename Dtype>
void LinearImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  const int in_channels = in.channels();
  const int in_height = in.rows;
  const int in_width = in.cols;

  // out uses the same number of channels as in, but uses floats
  out.create(in.size(), CV_32F | (0x18 & in.type()));

  // set up lookup arrays
  float scales[4], shifts[4];  
  if (param_.shift_size() == 1) {
    shifts[0] = shifts[1] = shifts[2] = shifts[3] = param_.shift(0);
  } else {
    for (int i = 0; i < param_.shift_size(); i++) {
	  shifts[i] = param_.shift(i);
	}
  }
  if (param_.scale_size() == 1) {
    scales[0] = scales[1] = scales[2] = scales[3] = param_.scale(0);
  } else {
    for (int i = 0; i < param_.scale_size(); i++) {
	  scales[i] = param_.scale(i);
	}
  }
  
  for (int h = 0; h < in_height; ++h) {
    // channel values are 1 byte wide (uchar)
	if (in.elemSize1() == 1) {
      const uchar* in_ptr = in.ptr<uchar>(h);
      float* out_ptr = out.ptr<float>(h);
      int index = 0;
      for (int w = 0; w < in_width; ++w) {
        for (int c = 0; c < in_channels; ++c) {
  	      out_ptr[index] = scales[c] * (in_ptr[index] - shifts[c]);
          //LOG(INFO) << "c: " << c << " h: " << h << " w: " << w << " index: " << index << " in_val: " << ((float)in_ptr[index]) << " out_val: " << out_ptr[index];
  	      index++;
        }
      }
	}  else if (in.elemSize1() == 4) {
      const float* in_ptr = in.ptr<float>(h);
      float* out_ptr = out.ptr<float>(h);
      int index = 0;
      for (int w = 0; w < in_width; ++w) {
        for (int c = 0; c < in_channels; ++c) {
  	      out_ptr[index] = scales[c] * (in_ptr[index] - shifts[c]);
          //LOG(INFO) << "c: " << c << " h: " << h << " w: " << w << " index: " << index << " in_val: " << ((float)in_ptr[index]) << " out_val: " << out_ptr[index];
  	      index++;
        }
      }
	}
  }
}

template <typename Dtype>
vector<int> LinearImageTransformer<Dtype>::InferOutputShape(const vector<int>& in_shape) {
  CHECK_GE(in_shape.size(), 3) << "Must know the number of channels";
  int in_channels = in_shape[in_shape.size() - 3];
  if (param_.has_shift_file()) {
  	// do nothing

  } else {
	  if (param_.shift_size() != 1 && param_.shift_size() != in_channels) {
		CHECK(0) << "Number of shifts is " << param_.shift_size() << " but number of channels is " <<
		  in_channels;
	  }
  }
  if (param_.scale_size() != 1 && param_.scale_size() != in_channels) {
    CHECK(0) << "Number of scales is " << param_.scale_size() << " but number of channels is " <<
	  in_channels;
  }
  return in_shape;
}

INSTANTIATE_CLASS(LinearImageTransformer);

}  // namespace caffe
