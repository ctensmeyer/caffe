#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/tukey_biweight_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class TukeyBiweightLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TukeyBiweightLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);

	Dtype* data = blob_bottom_data_->mutable_cpu_data();
	Dtype* label = blob_bottom_label_->mutable_cpu_data();
	
	// Make sure that no values fall at the 
	// discontinuity and break the gradient checker
	for (int i = 0; i < 50; i++) {
	  Dtype diff = data[i] - label[i];
	  if (diff > 2.9) {
	    //data[i] += 0.2;
	  }
	  if (diff < -2.9) {
	    //label[i] -= 0.2;
	  }
	  LOG(INFO) << i << ":\t" << data[i] << "\t" << label[i] << "\t" << diff;
	}
  }
  virtual ~TukeyBiweightLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

//TYPED_TEST_CASE(TukeyBiweightLossLayerTest, TestDtypesAndDevices);
TYPED_TEST_CASE(TukeyBiweightLossLayerTest, ::testing::Types<CPUDevice<double> >);


/*
TYPED_TEST(TukeyBiweightLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_tukey_biweight_param()->set_c(3);
  TukeyBiweightLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(TukeyBiweightLossLayerTest, TestGradientScale) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_tukey_biweight_param()->set_c(3);
  layer_param.mutable_tukey_biweight_param()->add_scale(1);
  layer_param.mutable_tukey_biweight_param()->add_scale(1.5);
  layer_param.mutable_tukey_biweight_param()->add_scale(2);
  layer_param.mutable_tukey_biweight_param()->add_scale(2.5);
  layer_param.mutable_tukey_biweight_param()->add_scale(3);
  TukeyBiweightLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
*/

TYPED_TEST(TukeyBiweightLossLayerTest, TestOutlierSlope) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_tukey_biweight_param()->set_c(3);
  layer_param.mutable_tukey_biweight_param()->set_outlier_slope(0.5);
  TukeyBiweightLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
