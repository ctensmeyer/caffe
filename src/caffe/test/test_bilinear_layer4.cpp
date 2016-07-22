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
#define OUT_HEIGHT 4
#define OUT_WIDTH 5

namespace caffe {

template <typename TypeParam>
class BilinearInterpolationLayerTest4 : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BilinearInterpolationLayerTest4()
      : blob_bottom_data_(new Blob<Dtype>(1, 1, 3, 3)),
        blob_bottom_dims_(new Blob<Dtype>(1, 1, 1, 2)),
        blob_top_data_(new Blob<Dtype>(1, 1, OUT_HEIGHT, OUT_WIDTH)),
		expected_result_(new Blob<Dtype>(1, 1, OUT_HEIGHT, OUT_WIDTH)) {
    // fill the values
	Dtype* bottom_data = blob_bottom_data_->mutable_cpu_data();
	for (int i = 0; i < 9; i++) {
      bottom_data[i] = MULT * i;
	}

	Dtype* dims = blob_bottom_dims_->mutable_cpu_data();
	dims[0] = 4; // height
	dims[1] = 5; // width

	blob_bottom_vec_.push_back(blob_bottom_data_);
	blob_bottom_vec_.push_back(blob_bottom_dims_);

	Dtype* top_diff = blob_top_data_->mutable_cpu_diff();
	Dtype* expected = expected_result_->mutable_cpu_data();
	for (int i = 0; i < OUT_HEIGHT * OUT_WIDTH; i++) {
	  top_diff[i] = MULT;
	  expected[i] = MULT * ((i - i / OUT_WIDTH) * 0.5);
	}
	blob_top_vec_.push_back(blob_top_data_);

  }
  virtual ~BilinearInterpolationLayerTest4() {
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
	for (int i = 0; i < OUT_HEIGHT * OUT_WIDTH; i++) {
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
	for (int i = 0; i < 9; i++) {
	  Dtype expected = MULT * (2 + ((i % 3 == 1) ? (2./3.) : 0));
	  //DLOG(ERROR) << "i/result/expected " << i << " " << result[i] << " " << expected;
      EXPECT_NEAR(result[i], expected, kErrorMargin);
	}
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_dims_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const expected_result_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BilinearInterpolationLayerTest4, TestDtypesAndDevices);

TYPED_TEST(BilinearInterpolationLayerTest4, TestForward) {
  this->TestForward();
}

TYPED_TEST(BilinearInterpolationLayerTest4, TestBackward) {
  this->TestBackward();
}

TYPED_TEST(BilinearInterpolationLayerTest4, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BilinearInterpolationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1702);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
