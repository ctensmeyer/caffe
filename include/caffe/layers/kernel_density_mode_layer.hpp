#ifndef CAFFE_KERNELDENSITYMODE_LAYER_HPP_
#define CAFFE_KERNELDENSITYMODE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {
template<typename Dtype>
class KernelDensityModeLayer : public Layer<Dtype> {

 public:
  explicit KernelDensityModeLayer(const LayerParameter& param) 
  	: Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  	const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
  	const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "KernelDensityMode"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinNumTopBlobs() const { return 1; }
  virtual inline int MaxNumTopBlobs() const { return 2; }

  // public to facilitate unit tests
  virtual void GradientAscent(const Dtype* grid, const int height, const int width, 
    double initial_h, double initial_w, double& center_h, double& center_w,
	double initial_step_size, int initial_line_searchiters, int max_iters);
  virtual double LineSearch(const Dtype* grid, const int height, const int width, 
    double h, double w, double dh, double dw, int num_steps, double max_step, bool backwards);
  virtual double GridVal(const Dtype* grid, const int height, const int width, 
    double h, double w);
  virtual void dGridVal(const Dtype* grid, const int height, const int width, 
    double h, double w, double& dh, double& dw);
  virtual double KernelVal(double dist);
  virtual double dKernelVal(double dist);
  virtual void dOutdGrid(double h_star, double w_star, int h, int w, 
    double& dh_star_dg, double& dw_star_dg);
  virtual void FiniteDiff(Dtype* grid, int height, int width, double eps,
      double h_star, double w_star, int h, int w, double& dh_star_dg, double& dw_star_dg);
  virtual double SumMagdGridVal(const Dtype* grid, const int height, const int width, double h, double w);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> conv_top_;
  vector<Blob<Dtype>*> conv_bottom_vec_, conv_top_vec_;
  ConvolutionLayer<Dtype>* conv_layer_;
  Dtype radius_, grad_thresh_;
  bool finite_diff_;
};


}  // namespace caffe

#endif 
