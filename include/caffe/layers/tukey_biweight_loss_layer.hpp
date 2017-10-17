#ifndef CAFFE_TUKEYLOSS_LAYER_HPP_
#define CAFFE_TUKEYLOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class TukeyBiweightLossLayer : public LossLayer<Dtype> {
 public:
  explicit TukeyBiweightLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TukeyBiweightLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype c_;
  bool normalize_;

};


}  // namespace caffe

#endif 
