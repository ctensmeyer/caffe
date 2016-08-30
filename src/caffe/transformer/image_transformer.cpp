
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
ImageTransformer<Dtype>* CreateImageTransformer(ImageTransformationParameter param) {
  const unsigned int rng_seed = (param.has_rng_seed()) ? param.rng_seed() : 0;
  vector<ImageTransformer<Dtype>*>* transformers = new vector<ImageTransformer<Dtype>*>();
  for (int i = 0; i < param.params_size(); i++) {
    ProbImageTransformParameter prob_param = param.params(i);
    vector<ImageTransformer<Dtype>*>* prob_transformers = new vector<ImageTransformer<Dtype>*>();
	vector<float> weights;

    float weight;

	// Resize
	for (int j = 0; j < prob_param.resize_params_size(); j++) {
	  ResizeTransformParameter resize_param = prob_param.resize_params(j); 
	  if (j < prob_param.resize_prob_weights_size()) {
	    weight = prob_param.resize_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new ResizeImageTransformer<Dtype>(resize_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}

    // Linear
	for (int j = 0; j < prob_param.linear_params_size(); j++) {
	  LinearTransformParameter linear_param = prob_param.linear_params(j); 
	  if (j < prob_param.linear_prob_weights_size()) {
	    weight = prob_param.linear_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new LinearImageTransformer<Dtype>(linear_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}
	// Crop
	for (int j = 0; j < prob_param.crop_params_size(); j++) {
	  CropTransformParameter crop_param = prob_param.crop_params(j); 
	  if (j < prob_param.crop_prob_weights_size()) {
	    weight = prob_param.crop_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new CropImageTransformer<Dtype>(crop_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}
	// Reflect
	for (int j = 0; j < prob_param.reflect_params_size(); j++) {
	  ReflectTransformParameter reflect_param = prob_param.reflect_params(j); 
	  if (j < prob_param.reflect_prob_weights_size()) {
	    weight = prob_param.reflect_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new ReflectImageTransformer<Dtype>(reflect_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}
	// GaussNoise
	for (int j = 0; j < prob_param.gauss_noise_params_size(); j++) {
	  GaussNoiseTransformParameter gauss_noise_param = prob_param.gauss_noise_params(j); 
	  if (j < prob_param.gauss_noise_prob_weights_size()) {
	    weight = prob_param.gauss_noise_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new GaussNoiseImageTransformer<Dtype>(gauss_noise_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}
	// Rotate
	for (int j = 0; j < prob_param.rotate_params_size(); j++) {
	  RotateTransformParameter rotate_param = prob_param.rotate_params(j); 
	  if (j < prob_param.rotate_prob_weights_size()) {
	    weight = prob_param.rotate_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new RotateImageTransformer<Dtype>(rotate_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}
	// Shear
	for (int j = 0; j < prob_param.shear_params_size(); j++) {
	  ShearTransformParameter shear_param = prob_param.shear_params(j); 
	  if (j < prob_param.shear_prob_weights_size()) {
	    weight = prob_param.shear_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new ShearImageTransformer<Dtype>(shear_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}
	// GaussBlur
	for (int j = 0; j < prob_param.gauss_blur_params_size(); j++) {
	  GaussBlurTransformParameter gauss_blur_param = prob_param.gauss_blur_params(j); 
	  if (j < prob_param.gauss_blur_prob_weights_size()) {
	    weight = prob_param.gauss_blur_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new GaussBlurImageTransformer<Dtype>(gauss_blur_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}
	// UnsharpMask
	for (int j = 0; j < prob_param.unsharp_mask_params_size(); j++) {
	  UnsharpMaskTransformParameter unsharp_mask_param = prob_param.unsharp_mask_params(j); 
	  if (j < prob_param.unsharp_mask_prob_weights_size()) {
	    weight = prob_param.unsharp_mask_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new UnsharpMaskImageTransformer<Dtype>(unsharp_mask_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}
	// Perspective
	for (int j = 0; j < prob_param.perspective_params_size(); j++) {
	  PerspectiveTransformParameter perspective_param = prob_param.perspective_params(j); 
	  if (j < prob_param.perspective_prob_weights_size()) {
	    weight = prob_param.perspective_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new PerspectiveImageTransformer<Dtype>(perspective_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}
	// ColorJitter
	for (int j = 0; j < prob_param.color_jitter_params_size(); j++) {
	  ColorJitterTransformParameter color_jitter_param = prob_param.color_jitter_params(j); 
	  if (j < prob_param.color_jitter_prob_weights_size()) {
	    weight = prob_param.color_jitter_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new ColorJitterImageTransformer<Dtype>(color_jitter_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}
	// ElasticDeformation
	for (int j = 0; j < prob_param.elastic_deformation_params_size(); j++) {
	  ElasticDeformationTransformParameter elastic_deformation_param = prob_param.elastic_deformation_params(j); 
	  if (j < prob_param.elastic_deformation_prob_weights_size()) {
	    weight = prob_param.elastic_deformation_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new ElasticDeformationImageTransformer<Dtype>(elastic_deformation_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}
	// ZeroBorder
	for (int j = 0; j < prob_param.zero_border_params_size(); j++) {
	  ZeroBorderTransformParameter zero_border_param = prob_param.zero_border_params(j); 
	  if (j < prob_param.zero_border_prob_weights_size()) {
	    weight = prob_param.zero_border_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new ZeroBorderImageTransformer<Dtype>(zero_border_param);
	  transformer->InitRand(rng_seed);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}

    ImageTransformer<Dtype>* prob_transformer = new ProbImageTransformer<Dtype>(prob_transformers, weights);
	prob_transformer->InitRand(rng_seed);
	transformers->push_back(prob_transformer);
  }
  ImageTransformer<Dtype>* seq_transformer = new SequenceImageTransformer<Dtype>(transformers);
  seq_transformer->InitRand(rng_seed);
  return seq_transformer;
}

// instantiate template function
template ImageTransformer<float>* CreateImageTransformer(ImageTransformationParameter param);
template ImageTransformer<double>* CreateImageTransformer(ImageTransformationParameter param);

template <typename Dtype>
void ImageTransformer<Dtype>::InitRand(unsigned int seed) {
  const unsigned int rng_seed = (seed) ? seed : caffe_rng_rand();
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
void ImageTransformer<Dtype>::RandFloat(const int n, const float min, const float max, float* out) {
  CHECK(rng_);
  CHECK_GE(max, min);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  boost::uniform_real<float> random_distribution(min, caffe_nextafter<float>(max));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<float> >
      variate_generator(rng, random_distribution);
  for (int i = 0; i < n; i++) {
    out[i] = variate_generator();
  }
}

template <typename Dtype>
void ImageTransformer<Dtype>::RandFloat(const int n, const double min, const double max, double* out) {
  CHECK(rng_);
  CHECK_GE(max, min);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  boost::uniform_real<double> random_distribution(min, caffe_nextafter<double>(max));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<double> >
      variate_generator(rng, random_distribution);
  for (int i = 0; i < n; i++) {
    out[i] = variate_generator();
  }
}

template <typename Dtype>
void ImageTransformer<Dtype>::RandGauss(const int n, const Dtype mean, const Dtype std_dev, Dtype* out) {
  CHECK(this->rng_);
  CHECK_GE(n, 0);
  CHECK(out);
  CHECK_GT(std_dev, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(this->rng_->generator());

  boost::normal_distribution<Dtype> random_distribution(mean, std_dev);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(rng, random_distribution);
  for (int i = 0; i < n; ++i) {
    out[i] = variate_generator();
  }
}

template <typename Dtype>
int ImageTransformer<Dtype>::GetInterpolation(Interpolation inter) {
  int interpolation = -1;
  switch (inter) {
    case INTER_NEAREST:
	  interpolation = cv::INTER_NEAREST;
	  break;
    case INTER_LINEAR:
	  interpolation = cv::INTER_LINEAR;
	  break;
    case INTER_AREA:
	  interpolation = cv::INTER_AREA;
	  break;
    case INTER_CUBIC:
	  interpolation = cv::INTER_CUBIC;
	  break;
    case INTER_LANCZOS4:
	  interpolation = cv::INTER_LANCZOS4;
	  break;
	default:
	  interpolation = cv::INTER_LINEAR;
	  break;
  }
  return interpolation;
}

template <typename Dtype>
int ImageTransformer<Dtype>::GetBorderMode(BorderMode mode) {
  int border_mode = -1;
  switch (mode) {
    case BORDER_CONSTANT:
	  border_mode = cv::BORDER_CONSTANT;
	  break;
    case BORDER_REFLECT:
	  border_mode = cv::BORDER_REFLECT;
	  break;
    case BORDER_REFLECT_101:
	  border_mode = cv::BORDER_REFLECT_101;
	  break;
    case BORDER_WRAP:
	  border_mode = cv::BORDER_WRAP;
	  break;
    case BORDER_REPLICATE:
	  border_mode = cv::BORDER_REPLICATE;
	  break;
	default:
	  border_mode = cv::BORDER_CONSTANT;
	  break;
  }
  return border_mode;
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
  ImageTransformer<Dtype>::SampleTransformParams(in_shape);
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
  ImageTransformer<Dtype>::SampleTransformParams(in_shape);
  SampleIdx();

  CHECK_GE(cur_idx_, 0) << "cur_idx_ is not initialized";
  CHECK_LT(cur_idx_, probs_.size()) << "cur_idx_ is too big: " << cur_idx_ << " vs. " << probs_.size();

  (*transformers_)[cur_idx_]->SampleTransformParams(in_shape);
}

template <typename Dtype>
void ProbImageTransformer<Dtype>::SampleIdx() {
  Dtype rand; 
  this->RandFloat(1, 0, 1, &rand);
  Dtype cum_prob = 0;
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
