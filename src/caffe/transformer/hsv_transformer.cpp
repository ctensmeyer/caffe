
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

  const uchar* in_ptr = in.ptr<uchar>(0);
  const uchar* out_ptr = out.ptr<uchar>(0);
  LOG(INFO) << "BGR: (" << ((int)in_ptr[0]) << "," << ((int)in_ptr[1]) << "," << ((int)in_ptr[2]) << ")  HSV: (" << ((int)out_ptr[0]) << "," << ((int)out_ptr[1]) << "," << ((int)out_ptr[2]) << ")";
}

INSTANTIATE_CLASS(HSVImageTransformer);

}  // namespace caffe
