#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PowCenterOfMassLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PowCenterOfMassLayerTest() : 
    bottom_(new Blob<Dtype>(2, 2, 5, 5)),
    top_(new Blob<Dtype>(2, 2, 2, 1)) {

    FillerParameter filler_param;
    filler_param.set_min(0.1);
    filler_param.set_max(10.0);
    UniformFiller<Dtype> filler(filler_param);

/*
    Dtype* bottom = bottom_->mutable_cpu_data();
	for (int idx=0; idx < 9; idx++) {
	  bottom[idx] = 1 + idx;
	}
*/

    //filler.Fill(this->bottom_);

    blob_bottom_vec_.push_back(bottom_);
    blob_top_vec_.push_back(top_);

  }
  virtual ~PowCenterOfMassLayerTest() {
    delete bottom_;
    delete top_;
  }

  Blob<Dtype>* const bottom_;
  Blob<Dtype>* const top_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(PowCenterOfMassLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(PowCenterOfMassLayerTest, ::testing::Types<CPUDevice<float> >);


TYPED_TEST(PowCenterOfMassLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_center_param()->set_pow(0.5);
  PowCenterOfMassLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(PowCenterOfMassLayerTest, TestGradientSquared) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_center_param()->set_pow(2);
  PowCenterOfMassLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}


TYPED_TEST(PowCenterOfMassLayerTest, TestGradientNorm) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_center_param()->set_normalize(true);
  PowCenterOfMassLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(PowCenterOfMassLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_center_param()->set_pow(1);

  PowCenterOfMassLayer<Dtype> pow_layer(layer_param);
  pow_layer.LayerSetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  pow_layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  pow_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->top_->cpu_data();
  Dtype pow_y = top_data[0];
  Dtype pow_x = top_data[1];

  CenterOfMassLayer<Dtype> com_layer(layer_param);
  com_layer.LayerSetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  com_layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  com_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  top_data = this->top_->cpu_data();
  Dtype com_y = top_data[0];
  Dtype com_x = top_data[1];

  Dtype kErrorMargin = 0.001;
  EXPECT_NEAR(pow_y, com_y, kErrorMargin);
  EXPECT_NEAR(pow_x, com_x, kErrorMargin);
}

}  // namespace caffe
