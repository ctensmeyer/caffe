

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/random.hpp>

#include <string>
#include <vector>
#include <math.h>

#include "caffe/image_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void ShearImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  float shear_angle = this->RandFloat(-param_.max_shear_angle(), param_.max_shear_angle()); 
  float shear_factor = tan(shear_angle * 3.14159265 / 180.0);
  // out uses the same number of channels as in, but uses floats
  out.create(in.size(), CV_32F | (0x18 & in.type()));

  cv::Mat shear(2, 3, CV_32F);

  // no scaling
  shear.at<float>(0,0) = 1;
  shear.at<float>(1,1) = 1;

  if (this->RandInt(2)) {
    // shear in x
    shear.at<float>(0,1) = shear_factor;
    shear.at<float>(1,0) = 0;
  } else {
    // shear in y
    shear.at<float>(0,1) = 0;
    shear.at<float>(1,0) = shear_factor;
  }

  // no translation
  shear.at<float>(0,2) = 0;
  shear.at<float>(1,2) = 0;
  cv::warpAffine(in, out, shear, in.size());
}

INSTANTIATE_CLASS(ShearImageTransformer);

}  // namespace caffe
