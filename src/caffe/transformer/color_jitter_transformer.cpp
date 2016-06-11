

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
void ColorJitterImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  const int in_channels = in.channels();
  const int in_height = in.rows;
  const int in_width = in.cols;
  // out is same dims as in, but must be float
  out.create(in.size(), CV_32F | (0x18 & in.type()));

  vector<float> channel_vals;
  for (int c = 0; c < in_channels; c++) {
  	Dtype channel_val;
    this->RandGauss(1, 0, this->param_.sigma(), &channel_val);

    channel_vals.push_back((float) channel_val);
  }


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
  	      out_ptr[index] = in_ptr[index] + channel_vals[c];
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
  	      out_ptr[index] = in_ptr[index] + channel_vals[c];
          //DLOG(INFO) << "c: " << c << " h: " << h << " w: " << w << " index: " << index << " in_val: " << ((float)in_ptr[index]) << " + " << rand_data[index] << " = " << out_ptr[index];
  	      index++;
        }
      }
	}
  }
}

INSTANTIATE_CLASS(ColorJitterImageTransformer);

}  // namespace caffe
