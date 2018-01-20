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
class MedianCenterOfMassLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MedianCenterOfMassLayerTest() : 
    bottom_(new Blob<Dtype>(2, 2, 3, 3)),
    top_(new Blob<Dtype>(2, 2, 2, 1)),
    bottom2_(new Blob<Dtype>(1, 1, 20, 20)),
    top2_(new Blob<Dtype>(1, 1, 2, 1)) {

    FillerParameter filler_param;
    filler_param.set_min(0.1);
    filler_param.set_max(10.0);
    UniformFiller<Dtype> filler(filler_param);

    filler.Fill(this->bottom_);
/*
    Dtype* bottom = bottom_->mutable_cpu_data();
	for (int idx=0; idx < 9; idx++) {
	  bottom[idx] = idx;
	}
*/

    blob_bottom_vec_.push_back(bottom_);
    blob_top_vec_.push_back(top_);


	Dtype* data = bottom2_->mutable_cpu_data();

	for (int i = 0; i < 400; i++) {
	  data[i] = 0;
	}

	for (int i = 8; i < 12; i++) {
	  data[20 * i + 10] = 10 + (i - 8);
	  data[20 * i + 11] = 10 + (i - 8);
	  data[20 * i + 14] = 10 + (i - 8);
	  data[20 * i + 15] = 20 + (i - 8);
	}

    blob_bottom_vec2_.push_back(bottom2_);
    blob_top_vec2_.push_back(top2_);

  }
  virtual ~MedianCenterOfMassLayerTest() {
    delete bottom_;
    delete bottom2_;
    delete top_;
    delete top2_;
  }

  Blob<Dtype>* const bottom_;
  Blob<Dtype>* const top_;

  Blob<Dtype>* const bottom2_;
  Blob<Dtype>* const top2_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec2_;
  vector<Blob<Dtype>*> blob_top_vec2_;

};

TYPED_TEST_CASE(MedianCenterOfMassLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(MedianCenterOfMassLayerTest, ::testing::Types<CPUDevice<float> >);


TYPED_TEST(MedianCenterOfMassLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MedianCenterOfMassLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}


TYPED_TEST(MedianCenterOfMassLayerTest, TestGradientNorm) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_center_param()->set_normalize(true);
  MedianCenterOfMassLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(MedianCenterOfMassLayerTest, TestGap) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MedianCenterOfMassLayer<Dtype> layer(layer_param);
  layer.LayerSetUp(this->blob_bottom_vec2_, this->blob_top_vec2_);
  layer.Reshape(this->blob_bottom_vec2_, this->blob_top_vec2_);
  layer.Forward(this->blob_bottom_vec2_, this->blob_top_vec2_);

  const Dtype* top_data = this->top2_->cpu_data();
  Dtype out_y = top_data[0];
  Dtype out_x = top_data[1];

  Dtype kErrorMargin = 0.00001;
  EXPECT_NEAR(out_y, 10.137931, kErrorMargin);
  EXPECT_NEAR(out_x, 13.304348, kErrorMargin);

  // check with a kink around 0 because if the CoM falls in a region of 0s
  // the finite differencing fundamentally changes where the median falls
  // because when there is exactly 0, there is a range of points that satisfy
  // the condition that the mass to the left and right are equal.
  // this layer uses a somewhat arbitrary way of picking the output in this case
  // anyway, the gradient for those 0s should be 0
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.1);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
