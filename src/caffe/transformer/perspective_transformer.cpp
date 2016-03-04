

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
  Dtype sigma = this->RandFloat(0, param_.max_sigma()); 
  // out uses the same number of channels as in, but uses floats
  out.create(in.size(), CV_32F | (0x18 & in.type()));

  cv::Point2f src[4];
  src[0] = cv::Point2f(0,0);
  src[1] = cv::Point2f(1,0);
  src[2] = cv::Point2f(1,1);
  src[3] = cv::Point2f(0,1);

  Dtype random[8];
  this->RandGauss(8, (Dtype)0, sigma, random);
  cv::Point2f dst[4];
  dst[0] = cv::Point2f(0 + random[0], 0 + random[1]);
  dst[1] = cv::Point2f(1 + random[2], 0 + random[3]);
  dst[2] = cv::Point2f(1 + random[4], 1 + random[5]);
  dst[3] = cv::Point2f(0 + random[6], 1 + random[7]);

  cv::Mat perspective = cv::getPerspectiveTransform(src, dst);
  cv::warpPerspective(in, out, perspective, in.size());
}

INSTANTIATE_CLASS(PerspectiveImageTransformer);

}  // namespace caffe
