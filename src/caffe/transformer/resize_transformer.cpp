
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
ResizeImageTransformer<Dtype>::ResizeImageTransformer(const ResizeTransformParameter& resize_param) : 
	ImageTransformer<Dtype>(), param_(resize_param) {
  ValidateParam();
}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::ValidateParam() {
  int num_groups = 0;
  if (param_.width_size()) {
    CHECK(param_.height_size()) << "If width is specified, height must as well";
	CHECK_GT(param_.width(0), 0) << "width must be positive";
	CHECK_GT(param_.height(0), 0) << "height must be positive";

	if (param_.width_size() > 1) {
	  CHECK_GE(param_.width(1), param_.width(0)) << "width upper bound < lower bound";
	}
	if (param_.height_size() > 1) {
	  CHECK_GE(param_.height(1), param_.height(0)) << "height upper bound < lower bound";
	}
	num_groups++;
  }
  if (param_.size_size()) {
	CHECK_GT(param_.size(0), 0) << "Size must be positive";

	if (param_.size_size() > 1) {
	  CHECK_GE(param_.size(1), param_.size(0)) << "size upper bound < lower bound";
	}
	num_groups++;
  }
  if (param_.width_perc_size()) {
    CHECK(param_.height_perc_size()) << "If width_perc is specified, height_perc must as well";
	CHECK_GT(param_.width_perc(0), 0) << "width_perc must be positive";
	CHECK_GT(param_.height_perc(0), 0) << "height_perc must be positive";

	if (param_.width_perc_size() > 1) {
	  CHECK_GE(param_.width_perc(1), param_.width_perc(0)) << "width_perc upper bound < lower bound";
	}
	if (param_.height_perc_size() > 1) {
	  CHECK_GE(param_.height_perc(1), param_.height_perc(0)) << "height_perc upper bound < lower bound";
	}
	num_groups++;
  }
  if (param_.size_perc_size()) {
	CHECK_GT(param_.size_perc(0), 0) << "Size must be positive";

	if (param_.size_perc_size() > 1) {
	  CHECK_GE(param_.size_perc(1), param_.size_perc(0)) << "size_perc upper bound < lower bound";
	}
	num_groups++;
  }

  if (num_groups == 0) {
    CHECK(0) << "No group of resize parameters were specified";
  }
  if (num_groups > 1) {
    CHECK(0) << "Multiple groups of resize parameters were specified";
  }

}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::SampleTransformParams(const vector<int>& in_shape) {
  ImageTransformers<Dtype>::SampleTransformParams(in_shape);
  CHECK_GE(in_shape.size(), 2);
  CHECK_LE(in_shape.size(), 4);
  int in_width = in_shape[in_shape.size() - 1];
  int in_height = in_shape[in_shape.size() - 2];

  if (param_.width_size()) {
    SampleFixedIndependent();
  } else if (param_.size_size()) {
    SampleFixedTied();
  } else if (param_.width_perc_size()) {
    SamplePercIndependent(in_width, in_height);
  } else if (param_.size_perc_size()) {
    SamplePercTied(in_width, in_height);
  } else {
    CHECK(0) << "Invalid resize param";
  }
}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::SamplePercIndependent(int in_width, int in_height) {
  if (param_.width_perc_size() == 1) {
    cur_width_ = (int) (param_.width_perc(0) * in_width);
  } else {
    cur_width_ = (int) (this->RandFloat(param_.width_perc(0), param_.width_perc(1)) * in_width);
  }
  if (param_.height_perc_size() == 1) {
    cur_height_ = (int) (param_.height_perc(0) * in_height);
  } else {
    cur_height_ = (int) (this->RandFloat(param_.height_perc(0), param_.height_perc(1)) * in_height);
  }
}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::SamplePercTied(int in_width, int in_height) {
  if (param_.size_perc_size() == 1) {
    cur_width_ = (int) (param_.size_perc(0) * in_width);
    cur_height_ = (int) (param_.size_perc(0) * in_height);
  } else {
    float perc = this->RandFloat(param_.size_perc(0), param_.size_perc(1));
    cur_width_ = (int) (perc *  in_width);
    cur_height_ = (int) (perc * in_height);
  }
}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::SampleFixedIndependent() {
  if (param_.width_size() == 1) {
    cur_width_ = param_.width(0);
  } else {
    cur_width_ = this->RandInt(param_.width(1) - param_.width(0) + 1) + param_.width(0);
  }
  if (param_.height_size() == 1) {
    cur_height_ = param_.height(0);
  } else {
    cur_height_ = this->RandInt(param_.height(1) - param_.height(0) + 1) + param_.height(0);
  }
}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::SampleFixedTied() {
  if (param_.size_size() == 1) {
    cur_width_ = cur_height_ = param_.size(0);
  } else {
    cur_width_ = cur_height_ = this->RandInt(param_.size(1) - param_.size(0) + 1) + param_.size(0);
  }
}

template <typename Dtype>
vector<int> ResizeImageTransformer<Dtype>::InferOutputShape(const vector<int>& in_shape) {
  CHECK_GE(in_shape.size(), 2);
  CHECK_LE(in_shape.size(), 4);

  vector<int> shape;
  for (int i = 0; i < in_shape.size() - 2; i++) {
    shape.push_back(in_shape[i]);
  }
  shape.push_back(cur_width_);
  shape.push_back(cur_height_);
  return shape;
}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  int interpolation;
  switch (param_.interpolation()) {
    case ResizeTransformParameter::INTER_NEAREST:
	  interpolation = cv::INTER_NEAREST;
	  break;
    case ResizeTransformParameter::INTER_LINEAR:
	  interpolation = cv::INTER_LINEAR;
	  break;
    case ResizeTransformParameter::INTER_AREA:
	  interpolation = cv::INTER_AREA;
	  break;
    case ResizeTransformParameter::INTER_CUBIC:
	  interpolation = cv::INTER_CUBIC;
	  break;
    case ResizeTransformParameter::INTER_LANCZOS4:
	  interpolation = cv::INTER_LANCZOS4;
	  break;
	default:
	  interpolation = cv::INTER_NEAREST;
	  break;
  }
  cv::Size size(cur_width_, cur_height_);
  cv::resize(in, out, size, 0, 0, interpolation);
}

INSTANTIATE_CLASS(ResizeImageTransformer);

}  // namespace caffe
