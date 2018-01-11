#ifndef CAFFE_SIGMOID_CE_LOSS_LAYER2_HPP_
#define CAFFE_SIGMOID_CE_LOSS_LAYER2_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {
template <typename Dtype>
class SigmoidCrossEntropyLoss2Layer : public LossLayer<Dtype> {
 public:
  explicit SigmoidCrossEntropyLoss2Layer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SigmoidCrossEntropyLoss2"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;

  // whether to normalize by the number of values (true) or by the number of instances (false)
  bool normalize_;
};

}  // namespace caffe

#endif 
