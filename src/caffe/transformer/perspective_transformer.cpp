

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
void PerspectiveImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  CHECK_GT(param_.max_sigma(), 0) << "Max Sigma must be positive";
  CHECK_GT(param_.max_sigma(), param_.min_sigma()) << "Max Sigma must be greater than Min Sigma";
  Dtype sigma;
  this->RandFloat(1, param_.min_sigma(), param_.max_sigma(), &sigma); 
  // out uses the same number of channels as in, but uses floats
  out.create(in.size(), CV_32F | (0x18 & in.type()));
  int interpolation = this->GetInterpolation(param_.interpolation());
  int border_mode = this->GetBorderMode(param_.border_mode());
  cv::Scalar border_val(param_.border_val());

  cv::Point2f src[4];
  src[0] = cv::Point2f(0,0);
  src[1] = cv::Point2f(1,0);
  src[2] = cv::Point2f(1,1);
  src[3] = cv::Point2f(0,1);

  cv::Point2f dst[4];
  if (this->param_.values_size() >= 8) {
	dst[0] = cv::Point2f(0 + this->param_.values(0), 0 + this->param_.values(1));
	dst[1] = cv::Point2f(1 + this->param_.values(2), 0 + this->param_.values(3));
	dst[2] = cv::Point2f(1 + this->param_.values(4), 1 + this->param_.values(5));
	dst[3] = cv::Point2f(0 + this->param_.values(6), 1 + this->param_.values(7));
  } else {
    Dtype random[8];
    this->RandGauss(8, (Dtype)0, sigma, random);
    dst[0] = cv::Point2f(0 + random[0], 0 + random[1]);
    dst[1] = cv::Point2f(1 + random[2], 0 + random[3]);
    dst[2] = cv::Point2f(1 + random[4], 1 + random[5]);
    dst[3] = cv::Point2f(0 + random[6], 1 + random[7]);
  }

  cv::Mat perspective = cv::getPerspectiveTransform(src, dst);
  cv::warpPerspective(in, out, perspective, in.size(), interpolation, border_mode, border_val);
}

INSTANTIATE_CLASS(PerspectiveImageTransformer);

}  // namespace caffe
