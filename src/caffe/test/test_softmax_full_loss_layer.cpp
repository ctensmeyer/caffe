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
#define BINS 5


namespace caffe {


template <typename TypeParam>
class SoftmaxFullLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxFullLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(NUM, BINS, 1, 1)),
        blob_bottom_target_probs_(new Blob<Dtype>(NUM, BINS, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
	/*
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
	*/
	blob_bottom_data_->mutable_cpu_data()[0] = -0.5;  // 0.05801221739799788
	blob_bottom_data_->mutable_cpu_data()[1] = 0;     // 0.09564597678455915
	blob_bottom_data_->mutable_cpu_data()[2] = 0.5;   // 0.15769355638159332
	blob_bottom_data_->mutable_cpu_data()[3] = 1;     // 0.25999272065868284
	blob_bottom_data_->mutable_cpu_data()[4] = 1.5;   // 0.428655528777167

	blob_bottom_target_probs_->mutable_cpu_data()[0] = 0.05;  
	blob_bottom_target_probs_->mutable_cpu_data()[1] = 0.15;
	blob_bottom_target_probs_->mutable_cpu_data()[2] = 0.0;
	blob_bottom_target_probs_->mutable_cpu_data()[3] = 0.3;
	blob_bottom_target_probs_->mutable_cpu_data()[4] = 0.5;

/*
    FillerParameter filler_param2;
	filler_param.set_min(0);
	filler_param.set_max(1);
	UniformFiller<Dtype> filler2(filler_param2);
	filler2.Fill(this->blob_bottom_target_probs_);

	// normalize randomly filled data
    for (int i = 0; i < NUM; ++i) {
	  Dtype sum = 0;
      for (int j = 0; j < BINS; ++j) {
        sum += blob_bottom_target_probs_->cpu_data()[i * BINS + j];
	  }
      for (int j = 0; j < BINS; ++j) {
        blob_bottom_target_probs_->mutable_cpu_data()[i * BINS + j] /= sum;
	  }
    }
*/

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_target_probs_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SoftmaxFullLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_target_probs_;
    delete blob_top_loss_;
  }

  void TestForward() {
    LayerParameter layer_param;
    layer_param.mutable_loss_param()->set_normalize(false);
    SoftmaxFullLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

	const Dtype result = this->blob_top_loss_->cpu_data()[0];
	const Dtype expected = 1.32210164583;
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(result, expected, kErrorMargin);
  }

  void TestBackward() {
    LayerParameter layer_param;
    layer_param.mutable_loss_param()->set_normalize(false);
    SoftmaxFullLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	vector<bool> propagate_down;
	propagate_down.push_back(true);
	layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

	const Dtype* result = blob_bottom_data_->cpu_diff();
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(result[0], 0.00801221739799788, kErrorMargin);
    EXPECT_NEAR(result[1], -0.05435402321544085, kErrorMargin);
    EXPECT_NEAR(result[2], 0.15769355638159332, kErrorMargin);
    EXPECT_NEAR(result[3], -0.04000727934131715, kErrorMargin);
    EXPECT_NEAR(result[4], -0.071344471222833, kErrorMargin);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_target_probs_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxFullLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxFullLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(SoftmaxFullLossLayerTest, TestBackward) {
  this->TestBackward();
}

TYPED_TEST(SoftmaxFullLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  SoftmaxFullLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxFullLossLayerTest, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  SoftmaxFullLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
