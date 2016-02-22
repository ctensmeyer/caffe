
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
  bool do_horz_reflect = (param_.horz_reflect_prob() >= this->RandFloat(0, 1));
  bool do_vert_reflect = (param_.vert_reflect_prob() >= this->RandFloat(0, 1));
  if (do_horz_reflect || do_vert_reflect) {
    int flip_code;
	if (do_horz_reflect && do_vert_reflect) {
	  flip_code = -1;
	} else if (do_horz_reflect) {
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

INSTANTIATE_CLASS(ReflectImageTransformer);

}  // namespace caffe
