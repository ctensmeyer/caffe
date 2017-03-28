

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
void OtsuColorJitterImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  const int in_channels = in.channels();
  const int in_height = in.rows;
  const int in_width = in.cols;

  CHECK_EQ(in_channels, 1) << "Otsu masked color jitter can only be applied to single channel images";
  CHECK_EQ(in.elemSize1(), 1) << "Otsu masked color jitter can only be applied to 8-bit images";

  cv::Mat mask;
  mask.create(in.size(), in.type());
  // out is same dims as in, but must be float
  out.create(in.size(), CV_32F | (0x18 & in.type()));

  if (this->param_.foreground()) {
    cv::threshold(in, mask, 0., 1., cv::THRESH_OTSU + cv::THRESH_BINARY_INV); 
  } else {
    cv::threshold(in, mask, 0., 1., cv::THRESH_OTSU + cv::THRESH_BINARY); 
  }

  Dtype val;
  this->RandGauss(1, this->param_.mean(), this->param_.sigma(), &val);

  for (int h = 0; h < in_height; ++h) {
    // channel values are 1 byte wide (uchar)
    const uchar* in_ptr = in.ptr<uchar>(h);
    const uchar* mask_ptr = mask.ptr<uchar>(h);
    float* out_ptr = out.ptr<float>(h);
    int index = 0;
    for (int w = 0; w < in_width; ++w) {
      out_ptr[index] = in_ptr[index] + mask_ptr[index] * val;
      //DLOG(INFO) << "c: " << c << " h: " << h << " w: " << w << " index: " << index << " in_val: " << ((float)in_ptr[index]) << " + " << rand_data[index] << " = " << out_ptr[index];
      index++;
    }
  }
}

INSTANTIATE_CLASS(OtsuColorJitterImageTransformer);

}  // namespace caffe
