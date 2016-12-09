
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
void HSVImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  cv::cvtColor(in, out, CV_BGR2HSV);
}

INSTANTIATE_CLASS(HSVImageTransformer);

}  // namespace caffe
