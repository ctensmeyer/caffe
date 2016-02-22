
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
void ReflectImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  if (reflect_h_ || reflect_v_) {
    int flip_code;
	if (reflect_h_ && reflect_v_) {
	  flip_code = -1;
	} else if (reflect_h_) {
	  flip_code = 0;
	} else {
	  flip_code = 1;
	}
	cv::flip(in, out, flip_code);
  } else {
    // no op
    out = in;
  }
}

template <typename Dtype>
void ReflectImageTransformer<Dtype>::SampleTransformParams(const vector<int>& in_shape) {
  reflect_h_ = this->RandFloat(0, 1) <= param_.horz_reflect_prob();
  reflect_v_ = this->RandFloat(0, 1) <= param_.vert_reflect_prob();
}

INSTANTIATE_CLASS(ReflectImageTransformer);

}  // namespace caffe
