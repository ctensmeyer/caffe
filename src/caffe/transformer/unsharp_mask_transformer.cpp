

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
void UnsharpMaskImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  CHECK_GT(param_.max_sigma(), 0) << "Max Sigma must be positive";
  CHECK_GT(param_.max_sigma(), param_.min_sigma()) << "Max Sigma must be greater than Min Sigma";
  CHECK_GE(param_.max_amount(), 0) << "Amount must be non-negative";
  CHECK_GE(param_.max_amount(), param_.min_amount()) << "Max Amount must be greater than Min Amount";
  Dtype sigma, amount;
  this->RandFloat(1, param_.min_sigma(), param_.max_sigma(), &sigma); 
  this->RandFloat(1, param_.min_amount(), param_.max_amount(), &amount); 
  int size = (int) (4 * sigma + 0.999);
  if (size % 2 == 0) {
  	 size++;
  }

  // out uses the same number of channels as in, but uses floats
  out.create(in.size(), CV_32F | (0x18 & in.type()));
  cv::Mat blurred;
  cv::GaussianBlur(in, blurred, cv::Size(size,size), sigma);
  cv::addWeighted(in, 1 + amount, blurred, -amount, 0, out);

}

INSTANTIATE_CLASS(UnsharpMaskImageTransformer);

}  // namespace caffe
