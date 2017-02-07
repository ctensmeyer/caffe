#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/graph_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class Graph4CEnergyLayerTest2 : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Graph4CEnergyLayerTest2() : 
    unary_(new Blob<Dtype>(10, 2, 2, 2)),
    pair_(new Blob<Dtype>(10, 8, 2, 2)),
    labels_(new Blob<Dtype>(10, 1, 2, 2)),
    energy_(new Blob<Dtype>(10, 1, 1, 1)) {

    FillerParameter filler_param;
    filler_param.set_min(0.0);
    filler_param.set_max(1.0);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(unary_);
    filler.Fill(pair_);

    blob_bottom_vec_.push_back(unary_);
    blob_bottom_vec_.push_back(pair_);

	Dtype* labels = labels_->mutable_cpu_data();
	for (int i = 0; i < labels_->count(); i++) {
	  int tmp;
	  caffe_rng_bernoulli(1, 0.5, &tmp);
	  labels[i] = (Dtype) tmp;
	}

    blob_bottom_vec_.push_back(labels_);

    blob_top_vec_.push_back(energy_);
  }
  virtual ~Graph4CEnergyLayerTest2() {
    delete unary_;
    delete pair_;
    delete labels_;
    delete energy_;
  }

  Blob<Dtype>* const unary_;
  Blob<Dtype>* const pair_;
  Blob<Dtype>* const labels_;
  Blob<Dtype>* const energy_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(Graph4CEnergyLayerTest2, TestDtypesAndDevices);


TYPED_TEST(Graph4CEnergyLayerTest2, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Graph4CEnergyLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 2e-2, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}


}  // namespace caffe
