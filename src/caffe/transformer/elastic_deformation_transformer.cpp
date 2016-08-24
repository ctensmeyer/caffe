
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
void ElasticDeformationImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  const int in_height = in.rows;
  const int in_width = in.cols;

  float sigma, alpha;
  sigma = this->param_.sigma();
  this->RandFloat(1, 0, this->param_.max_alpha(), &alpha);

  // out is same dims as in, but must be float
  out.create(in.size(), CV_32F | (0x18 & in.type()));
  int interpolation = this->GetInterpolation(param_.interpolation());
  cv::Scalar border_val(param_.border_val());

  vector<int> shape;
  shape.push_back(in_height);
  shape.push_back(in_width);
  dis_x_->Reshape(shape);
  dis_y_->Reshape(shape);
  this->RandFloat(in_height * in_width, -1 * alpha, alpha, dis_x_->mutable_cpu_data());
  this->RandFloat(in_height * in_width, -1 * alpha, alpha, dis_y_->mutable_cpu_data());

  cv::Mat mat_dis_x(in_height, in_width, CV_32FC1, dis_x_->mutable_cpu_data());
  cv::Mat mat_dis_y(in_height, in_width, CV_32FC1, dis_y_->mutable_cpu_data());
  /*
  for (int i = 0; i < 5; i++) {
    printf("values: %.2f\t%.2f\n", dis_x_->data_at(0, i, 0, 0), mat_dis_x.at<float>(0, i));
  }
  */

  cv::GaussianBlur(mat_dis_x, mat_dis_x, cv::Size(0, 0), sigma);
  cv::GaussianBlur(mat_dis_y, mat_dis_y, cv::Size(0, 0), sigma);

  for(int y = 0; y < in_height; y++) {
    for(int x = 0; x < in_width; x++) {
	  mat_dis_x.at<float>(y, x) += x;
	  mat_dis_y.at<float>(y, x) += y;
	}
  }

  // clip values to valid range
  cv::normalize(mat_dis_x, mat_dis_x, 0, in_width, cv::NORM_MINMAX);
  cv::normalize(mat_dis_y, mat_dis_y, 0, in_height, cv::NORM_MINMAX);

  // default bilinear interpoloation
  cv::remap(in, out, mat_dis_x, mat_dis_y, interpolation, cv::BORDER_CONSTANT, border_val);
}

INSTANTIATE_CLASS(ElasticDeformationImageTransformer);

}  // namespace caffe
