
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
void ImageTransformer<Dtype>::InitRand() {
  const unsigned int rng_seed = caffe_rng_rand();
  rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
int ImageTransformer<Dtype>::RandInt(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
float ImageTransformer<Dtype>::RandFloat(float min, float max) {
  CHECK(rng_);
  CHECK_GE(max, min);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  boost::uniform_real<float> random_distribution(min, caffe_nextafter<float>(max));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<float> >
      variate_generator(rng, random_distribution);
  return variate_generator();
}

template <typename Dtype>
void ImageTransformer<Dtype>::CVMatToArray(const cv::Mat& cv_img, Dtype* out) {
  int cv_channels = cv_img.channels();
  int cv_height = cv_img.rows;
  int cv_width = cv_img.cols;
  for (int h = 0; h < cv_height; ++h) {
    if (cv_img.elemSize1() == 1) {
	  // channel values are 1 byte wide (uchar)
      const uchar* ptr = cv_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < cv_width; ++w) {
        for (int c = 0; c < cv_channels; ++c) {
          int out_index = (c * cv_height + h) * cv_width + w;
	      //LOG(INFO) << "c: " << c << " h: " << h << " w: " << w << " out_index: " << out_index << " value: " << ((float)ptr[img_index]);
	  	out[out_index] = static_cast<Dtype> (ptr[img_index++]);
        }
      }
	} else if (cv_img.elemSize1() == 4) {
	  // channel values are 4 bytes wide (float)
      const float* ptr = cv_img.ptr<float>(h);
      int img_index = 0;
      for (int w = 0; w < cv_width; ++w) {
        for (int c = 0; c < cv_channels; ++c) {
          int out_index = (c * cv_height + h) * cv_width + w;
	      //LOG(INFO) << "c: " << c << " h: " << h << " w: " << w << " out_index: " << out_index << " value: " << ((float)ptr[img_index]);
	  	out[out_index] = static_cast<Dtype> (ptr[img_index++]);
        }
      }
	}
  }
}

template <typename Dtype>
void SequenceImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  cv::Mat one = in;
  cv::Mat two;
  int i;
  for (i = 0; i < transformers_->size(); i++) {
    ImageTransformer<Dtype>* transformer = (*transformers_)[i];
	if (i % 2 == 0) {
	  transformer->Transform(one, two);
	} else {
	  transformer->Transform(two, one);
	}
  }
  // assign based on which variable last acted as the output variable.
  if ( (i - 1) % 2 == 0) {
    out = two;
  } else {
    out = one;
  }
}

template <typename Dtype>
void SequenceImageTransformer<Dtype>::SampleTransformParams(const vector<int>& in_shape) {
  vector<int> shape = in_shape;
  for (int i = 0; i < transformers_->size(); i++) {
    ImageTransformer<Dtype>* transformer = (*transformers_)[i];
	transformer->SampleTransformParams(shape);
	shape = transformer->InferOutputShape(shape);
  }
}

template <typename Dtype>
vector<int> SequenceImageTransformer<Dtype>::InferOutputShape(const vector<int>& in_shape) {
  vector<int> shape = in_shape;
  for (int i = 0; i < transformers_->size(); i++) {
    ImageTransformer<Dtype>* transformer = (*transformers_)[i];
	shape = transformer->InferOutputShape(shape);
  }
  return shape;
}

template <typename Dtype>
ProbImageTransformer<Dtype>::ProbImageTransformer(vector<ImageTransformer<Dtype>*>* transformers, vector<float> weights) :
  transformers_(transformers), probs_(weights) {
  CHECK(transformers_);
  CHECK_EQ(transformers_->size(), weights.size()) << "Number of transformers and weights must be equal: " <<
    transformers_->size() << " vs. " << weights.size();
  CHECK_GT(transformers_->size(), 0) << "Number of transformers must be positive";
  float sum = 0;
  for (int i = 0; i < probs_.size(); i++) {
    sum += probs_[i];
  }
  for (int i = 0; i < probs_.size(); i++) {
    probs_[i] /= sum;
  }
}

template <typename Dtype>
void ProbImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  CHECK_GE(cur_idx_, 0) << "cur_idx_ is not initialized";
  CHECK_LT(cur_idx_, probs_.size()) << "cur_idx_ is too big: " << cur_idx_ << " vs. " << probs_.size();

  (*transformers_)[cur_idx_]->Transform(in, out);
}

template <typename Dtype>
vector<int> ProbImageTransformer<Dtype>::InferOutputShape(const vector<int>& in_shape) {
  CHECK_GE(cur_idx_, 0) << "cur_idx_ is not initialized";
  CHECK_LT(cur_idx_, probs_.size()) << "cur_idx_ is too big: " << cur_idx_ << " vs. " << probs_.size();

  return (*transformers_)[cur_idx_]->InferOutputShape(in_shape);
}

template <typename Dtype>
void ProbImageTransformer<Dtype>::SampleTransformParams(const vector<int>& in_shape) {
  SampleIdx();

  CHECK_GE(cur_idx_, 0) << "cur_idx_ is not initialized";
  CHECK_LT(cur_idx_, probs_.size()) << "cur_idx_ is too big: " << cur_idx_ << " vs. " << probs_.size();

  (*transformers_)[cur_idx_]->SampleTransformParams(in_shape);
}

template <typename Dtype>
void ProbImageTransformer<Dtype>::SampleIdx() {
  float rand = this->RandFloat(0,1);
  float cum_prob = 0;
  int i;
  for (i = 0; i < probs_.size(); i++) {
    cum_prob += probs_[i];
	if (cum_prob >= rand) {
	  break;
    }
  }
  if (i == probs_.size()) {
    i--;
  }
  cur_idx_ = i;
}

INSTANTIATE_CLASS(ImageTransformer);
INSTANTIATE_CLASS(SequenceImageTransformer);
INSTANTIATE_CLASS(ProbImageTransformer);

}  // namespace caffe
