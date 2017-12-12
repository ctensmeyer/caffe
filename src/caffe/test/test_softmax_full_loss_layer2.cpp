#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;
#define NUM 1
#define NUM2 4
#define BINS 5


namespace caffe {


template <typename TypeParam>
class SoftmaxFullLossLayerTest2 : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxFullLossLayerTest2()
      : blob_bottom_data_(new Blob<Dtype>(NUM, NUM2, BINS, 1)),
        blob_bottom_target_probs_(new Blob<Dtype>(NUM, NUM2, BINS, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values

    FillerParameter filler_param;
    filler_param.set_std(1);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    FillerParameter filler_param2;
	filler_param.set_min(0);
	filler_param.set_max(1);
	UniformFiller<Dtype> filler2(filler_param2);
	filler2.Fill(this->blob_bottom_target_probs_);

	// normalize randomly filled data
    for (int i = 0; i < NUM; ++i) {
	  for (int k = 0; k < NUM2; k++) {
	    Dtype sum = 0;
        for (int j = 0; j < BINS; ++j) {
          sum += blob_bottom_target_probs_->cpu_data()[(i * NUM2 + k) * BINS + j];
	    }
        for (int j = 0; j < BINS; ++j) {
          blob_bottom_target_probs_->mutable_cpu_data()[(i * NUM2 + k) * BINS + j] /= sum;
	    }
	  }
    }

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_target_probs_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SoftmaxFullLossLayerTest2() {
    delete blob_bottom_data_;
    delete blob_bottom_target_probs_;
    delete blob_top_loss_;
  }


  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_target_probs_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxFullLossLayerTest2, TestDtypesAndDevices);


TYPED_TEST(SoftmaxFullLossLayerTest2, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  layer_param.mutable_softmax_param()->set_axis(2);

  SoftmaxFullLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxFullLossLayerTest2, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  layer_param.mutable_softmax_param()->set_axis(2);
  SoftmaxFullLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
