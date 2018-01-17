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
class CenterOfMassLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CenterOfMassLayerTest() : 
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

	data[20 * 10 + 10] = 10;
	data[20 * 10 + 11] = 10;
	data[20 * 11 + 10] = 10;
	data[20 * 11 + 11] = 10;

	// outlier
	data[20 * 17 + 17] = 5;
    blob_bottom_vec2_.push_back(bottom2_);
    blob_top_vec2_.push_back(top2_);

  }
  virtual ~CenterOfMassLayerTest() {
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

TYPED_TEST_CASE(CenterOfMassLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(CenterOfMassLayerTest, ::testing::Types<GPUDevice<float> >);


TYPED_TEST(CenterOfMassLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CenterOfMassLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(CenterOfMassLayerTest, TestGradientNorm) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_center_param()->set_normalize(true);
  CenterOfMassLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(CenterOfMassLayerTest, TestIterations) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_center_param()->set_radius(3);
  layer_param.mutable_center_param()->set_num_iters(2);
  layer_param.mutable_center_param()->set_min_multiple_iters(0);
  CenterOfMassLayer<Dtype> layer(layer_param);
  layer.Forward(this->blob_bottom_vec2_, this->blob_top_vec2_);

  const Dtype* top_data = this->top2_->cpu_data();
  Dtype out_y = top_data[0];
  Dtype out_x = top_data[1];

  Dtype kErrorMargin = 0.001;
  EXPECT_NEAR(out_y, 10.5, kErrorMargin);
  EXPECT_NEAR(out_x, 10.5, kErrorMargin);
}


}  // namespace caffe
