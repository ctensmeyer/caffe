
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
void RotateImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  float angle = this->RandFloat(-param_.max_angle(), param_.max_angle()); 
  // out uses the same number of channels as in, but uses floats
  out.create(in.size(), CV_32F | (0x18 & in.type()));

  const int in_channels = in.channels();
  const int in_height = in.rows;
  const int in_width = in.cols;
  cv::Point2f pt(in_height / 2., in_width / 2.);

  cv::Mat rotate = cv::getRotationMatrix2D(pt, angle, 1.0);
  cv::warpAffine(in, out, rotate, in.size());
}

INSTANTIATE_CLASS(RotateImageTransformer);

}  // namespace caffe
