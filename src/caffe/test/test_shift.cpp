#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/shift_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ShiftLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ShiftLayerTest() : 
    bottom_(new Blob<Dtype>(2, 2, 5, 5)),
    top_(new Blob<Dtype>(2, 2, 5, 5)) {

    FillerParameter filler_param;
    filler_param.set_min(-10.0);
    filler_param.set_max(10.0);
    UniformFiller<Dtype> filler(filler_param);

    filler.Fill(this->bottom_);

    blob_bottom_vec_.push_back(bottom_);
    blob_top_vec_.push_back(top_);
  }
  virtual ~ShiftLayerTest() {
    delete bottom_;
    delete top_;
  }

  Blob<Dtype>* const bottom_;
  Blob<Dtype>* const top_;


  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

//TYPED_TEST_CASE(ShiftLayerTest, TestDtypesAndDevices);
TYPED_TEST_CASE(ShiftLayerTest, ::testing::Types<CPUDevice<float> >);


TYPED_TEST(ShiftLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_shift_param()->set_h_shift(0);
  layer_param.mutable_shift_param()->set_w_shift(0);
  ShiftLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(ShiftLayerTest, TestGradient2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_shift_param()->set_h_shift(1);
  layer_param.mutable_shift_param()->set_w_shift(-1);
  ShiftLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(ShiftLayerTest, TestGradient3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_shift_param()->set_h_shift(-3);
  layer_param.mutable_shift_param()->set_w_shift(2);
  ShiftLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
