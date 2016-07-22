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

#define MULT 1.0

namespace caffe {

template <typename TypeParam>
class BilinearInterpolationLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BilinearInterpolationLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1, 1, 2, 2)),
        blob_bottom_dims_(new Blob<Dtype>(1, 1, 1, 2)),
        blob_top_data_(new Blob<Dtype>(1, 1, 3, 4)),
		expected_result_(new Blob<Dtype>(1, 1, 3, 4)) {
    // fill the values
	Dtype* bottom_data = blob_bottom_data_->mutable_cpu_data();
	bottom_data[0] = MULT * 1; // (0, 0) for (h, w)
	bottom_data[1] = MULT * 2; // (0, 1)
	bottom_data[2] = MULT * 3; // (1, 0)
	bottom_data[3] = MULT * 4; // (1, 1)

	Dtype* dims = blob_bottom_dims_->mutable_cpu_data();
	dims[0] = 3; // height
	dims[1] = 4; // width

	blob_bottom_vec_.push_back(blob_bottom_data_);
	blob_bottom_vec_.push_back(blob_bottom_dims_);

	Dtype* top_diff = blob_top_data_->mutable_cpu_diff();
	for (int i = 0; i < 12; i++) {
	  top_diff[i] = MULT;
	}
	blob_top_vec_.push_back(blob_top_data_);

	Dtype* expected = expected_result_->mutable_cpu_data();
	expected[0] = MULT * 1.;
	expected[1] = MULT * 1.33333333333333;
	expected[2] = MULT * 1.66666666666666;
	expected[3] = MULT * 2;
	expected[4] = MULT * 2;
	expected[5] = MULT * 2.33333333333333;
	expected[6] = MULT * 2.66666666666666;
	expected[7] = MULT * 3;
	expected[8] = MULT * 3;
	expected[9] = MULT * 3.33333333333333;
	expected[10] = MULT * 3.6666666666666;
	expected[11] = MULT * 4;

  }
  virtual ~BilinearInterpolationLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_dims_;
    delete blob_top_data_;
	delete expected_result_;
  }

  void TestForward() {
    LayerParameter layer_param;
    BilinearInterpolationLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

	const Dtype* result = blob_top_data_->cpu_data();
	const Dtype* expected = expected_result_->cpu_data();
    const Dtype kErrorMargin = 1e-5;
	for (int i = 0; i < 12; i++) {
	  //DLOG(ERROR) << "i/result/expected " << i << " " << result[i] << " " << expected[i];
      EXPECT_NEAR(result[i], expected[i], kErrorMargin);
	}
  }

  void TestBackward() {
    LayerParameter layer_param;
    BilinearInterpolationLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	vector<bool> propagate_down;
	propagate_down.push_back(true);
	layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

	const Dtype* result = blob_bottom_data_->cpu_diff();
    const Dtype kErrorMargin = 1e-5;
	for (int i = 0; i < 4; i++) {
	  //DLOG(ERROR) << "i/result/expected " << i << " " << result[i] << " " << 3 * MULT;
      EXPECT_NEAR(result[i], 3.0 * MULT, kErrorMargin);
	}
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_dims_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const expected_result_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BilinearInterpolationLayerTest, TestDtypesAndDevices);

///*
TYPED_TEST(BilinearInterpolationLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(BilinearInterpolationLayerTest, TestBackward) {
  this->TestBackward();
}
//*/

///*
TYPED_TEST(BilinearInterpolationLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BilinearInterpolationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1702);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
//*/

}  // namespace caffe
