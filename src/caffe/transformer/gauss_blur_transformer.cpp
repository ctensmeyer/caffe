
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
void GaussBlurImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  CHECK_GT(param_.max_sigma(), 0) << "Max Sigma must be positive";
  float sigma = this->RandFloat(0, param_.max_sigma()); 
  int size = (int) (2 * sigma + 0.999);
  // out uses the same number of channels as in, but uses floats
  out.create(in.size(), CV_32F | (0x18 & in.type()));
  cv::GaussianBlur(in, out, cv::Size(size,size), sigma);
}

INSTANTIATE_CLASS(GaussBlurImageTransformer);

}  // namespace caffe
