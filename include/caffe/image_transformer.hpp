#ifndef CAFFE_IMAGE_TRANSFORMER_HPP
#define CAFFE_IMAGE_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


// TODO: verify if the width/height dimension order is correct
template <typename Dtype>
class ImageTransformer {
 public:
  explicit ImageTransformer() { num_sampled_params_ = 0;}
  virtual ~ImageTransformer() {}

  void InitRand();
  void InitRand(unsigned int seed);
  int RandInt(int n);
  float RandFloat(float min, float max);

  void CVMatToArray(const cv::Mat& cv_img, Dtype* out);
  virtual void Transform(const cv::Mat& in, cv::Mat& out) {}
  virtual vector<int> InferOutputShape(const vector<int>& in_shape) {return in_shape;}
  virtual void SampleTransformParams(const vector<int>& in_shape) { num_sampled_params_++; }
  virtual void PrintParams() { 
    DLOG(INFO) << "PrintParams (" << this << ") " << "Num Sampled: " << num_sampled_params_;
  }

 protected:
  shared_ptr<Caffe::RNG> rng_;
  int num_sampled_params_;
};

template <typename Dtype>
ImageTransformer<Dtype>* CreateImageTransformer(ImageTransformationParameter param);


template <typename Dtype>
class ResizeImageTransformer : public ImageTransformer<Dtype> {
 public:
  explicit ResizeImageTransformer(const ResizeTransformParameter& resize_param);
  virtual ~ResizeImageTransformer() {}

  virtual void Transform(const cv::Mat& in, cv::Mat& out);
  virtual vector<int> InferOutputShape(const vector<int>& in_shape);
  virtual void SampleTransformParams(const vector<int>& in_shape);
  virtual void PrintParams();

 protected:
  void ValidateParam();
  void SampleFixedIndependent();
  void SampleFixedTied();
  void SamplePercIndependent(int in_width, int in_height);
  void SamplePercTied(int in_width, int in_height);
  ResizeTransformParameter param_;
  int cur_width_, cur_height_;
};

template <typename Dtype>
class SequenceImageTransformer : public ImageTransformer<Dtype> {
 public:
  explicit SequenceImageTransformer(vector<ImageTransformer<Dtype>*>* transformers) :
    transformers_(transformers) {}
  virtual ~SequenceImageTransformer() { if (transformers_) delete transformers_; } 

  virtual void Transform(const cv::Mat& in, cv::Mat& out);
  virtual vector<int> InferOutputShape(const vector<int>& in_shape);
  virtual void SampleTransformParams(const vector<int>& in_shape);

 protected:
  vector<ImageTransformer<Dtype>*>* transformers_;
};

template <typename Dtype>
class ProbImageTransformer : public ImageTransformer<Dtype> {
 public:
  explicit ProbImageTransformer(vector<ImageTransformer<Dtype>*>* transformers, vector<float> weights);
  virtual ~ProbImageTransformer() { if (transformers_) delete transformers_; }

  virtual void Transform(const cv::Mat& in, cv::Mat& out);
  virtual vector<int> InferOutputShape(const vector<int>& in_shape);
  virtual void SampleTransformParams(const vector<int>& in_shape);

 protected:
  void SampleIdx();
  vector<ImageTransformer<Dtype>*>* transformers_;
  vector<float> probs_;
  int cur_idx_;
};

// TODO: implement file parameters
template <typename Dtype>
class LinearImageTransformer : public ImageTransformer<Dtype> {
 public:
  explicit LinearImageTransformer(LinearTransformParameter param) :
    param_(param) {};
  virtual ~LinearImageTransformer() {};

  virtual void Transform(const cv::Mat& in, cv::Mat& out);
  virtual vector<int> InferOutputShape(const vector<int>& in_shape);
  virtual void SampleTransformParams(const vector<int>& in_shape) {}

 protected:
  virtual void LoadShiftFile();
  virtual void ResizeShiftImage(const vector<int>& in_shape);
  LinearTransformParameter param_;
  cv::Mat* shift_image_original_;
  cv::Mat* shift_image_current_;
};

template <typename Dtype>
class CropImageTransformer : public ImageTransformer<Dtype> {
 public:
  explicit CropImageTransformer(CropTransformParameter param) :
    param_(param) {};
  virtual ~CropImageTransformer() {};

  virtual void Transform(const cv::Mat& in, cv::Mat& out);
  virtual vector<int> InferOutputShape(const vector<int>& in_shape);
  virtual void SampleTransformParams(const vector<int>& in_shape);
  virtual void PrintParams();

 protected:
  CropTransformParameter param_;
  void ValidateParam();
  void SampleFixedIndependent();
  void SampleFixedTied();
  void SamplePercIndependent(int in_width, int in_height);
  void SamplePercTied(int in_width, int in_height);
  int cur_height_, cur_width_;
};

template <typename Dtype>
class ReflectImageTransformer : public ImageTransformer<Dtype> {
 public:
  explicit ReflectImageTransformer(ReflectTransformParameter param) :
    param_(param) {};
  virtual ~ReflectImageTransformer() {};

  virtual void Transform(const cv::Mat& in, cv::Mat& out);
  virtual void SampleTransformParams(const vector<int>& in_shape);
  virtual void PrintParams();

 protected:
  ReflectTransformParameter param_;
  bool reflect_h_;
  bool reflect_v_;
};

template <typename Dtype>
class GaussNoiseImageTransformer : public ImageTransformer<Dtype> {
 public:
  explicit GaussNoiseImageTransformer(GaussNoiseTransformParameter param) :
    param_(param) { rand_mask_ = new Blob<Dtype>();};
  virtual ~GaussNoiseImageTransformer() {};

  void RandGauss(const int n, const Dtype mean, const Dtype std_dev, Dtype* out);
  virtual void Transform(const cv::Mat& in, cv::Mat& out);
  virtual void SampleTransformParams(const vector<int>& in_shape);
  virtual void PrintParams();

 protected:
  GaussNoiseTransformParameter param_;
  float cur_std_dev_;
  Blob<Dtype>* rand_mask_;
};

template <typename Dtype>
class RotateImageTransformer : public ImageTransformer<Dtype> {
 public:
  explicit RotateImageTransformer(RotateTransformParameter param) :
    param_(param) { };
  virtual ~RotateImageTransformer() {};

  virtual void Transform(const cv::Mat& in, cv::Mat& out);

 protected:
  RotateTransformParameter param_;
};

template <typename Dtype>
class ShearImageTransformer : public ImageTransformer<Dtype> {
 public:
  explicit ShearImageTransformer(ShearTransformParameter param) :
    param_(param) { };
  virtual ~ShearImageTransformer() {};

  virtual void Transform(const cv::Mat& in, cv::Mat& out);

 protected:
  ShearTransformParameter param_;
};

template <typename Dtype>
class GaussBlurImageTransformer : public ImageTransformer<Dtype> {
 public:
  explicit GaussBlurImageTransformer(GaussBlurTransformParameter param) :
    param_(param) { };
  virtual ~GaussBlurImageTransformer() {};

  virtual void Transform(const cv::Mat& in, cv::Mat& out);

 protected:
  GaussBlurTransformParameter param_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_TRANSFORMER_HPP_

