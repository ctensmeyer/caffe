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
class WeightedFMeasureLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  WeightedFMeasureLossLayerTest() : 
  /*
  	    blob_bottom_data_(new Blob<Dtype>(1, 1, 1, 2)),
        blob_bottom_label_(new Blob<Dtype>(1, 1, 1, 2)),
        blob_bottom_recall_weight_(new Blob<Dtype>(1, 1, 1, 2)),
        blob_bottom_precision_weight_(new Blob<Dtype>(1, 1, 1, 2)),
  */
  	    blob_bottom_data_(new Blob<Dtype>(10, 1, 5, 5)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 5, 5)),
        blob_bottom_recall_weight_(new Blob<Dtype>(10, 1, 5, 5)),
        blob_bottom_precision_weight_(new Blob<Dtype>(10, 1, 5, 5)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0.0);
    filler_param.set_max(1.0);
    UniformFiller<Dtype> filler(filler_param);

    filler.Fill(this->blob_bottom_data_);
	/*
	Dtype* input = blob_bottom_data_->mutable_cpu_data();
	input[0] = 0.3;
	input[1] = 0.8;
	*/
    blob_bottom_vec_.push_back(blob_bottom_data_);

    // Assume binary targets
    int count = blob_bottom_label_->count();
	Dtype* target = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < count; i++) {
	  if (i < 100) {
	    target[i] = i % 2;
	  } else {
	    target[i] = i % 2;
	    //target[i] = 0;
	  }
	}

    blob_bottom_vec_.push_back(blob_bottom_label_);

    filler.Fill(this->blob_bottom_recall_weight_);
	/*
	Dtype* recall_weight = blob_bottom_recall_weight_->mutable_cpu_data();
	recall_weight[0] = 1.0;
	recall_weight[1] = 1.0;
	*/
    blob_bottom_vec_.push_back(blob_bottom_recall_weight_);

    filler.Fill(this->blob_bottom_precision_weight_);
	/*
	Dtype* precision_weight = blob_bottom_precision_weight_->mutable_cpu_data();
	precision_weight[0] = 1.0;
	precision_weight[1] = 1.0;
	*/
    blob_bottom_vec_.push_back(blob_bottom_precision_weight_);

    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~WeightedFMeasureLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_recall_weight_;
    delete blob_bottom_precision_weight_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_recall_weight_;
  Blob<Dtype>* const blob_bottom_precision_weight_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(WeightedFMeasureLossLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(WeightedFMeasureLossLayerTest, ::testing::Types<CPUDevice<float> >);


TYPED_TEST(WeightedFMeasureLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WeightedFmeasureLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 2e-2, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}


/*
TYPED_TEST(WeightedFMeasureLossLayerTest, TestMarginGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_weighted_fmeasure_loss_param()->set_margin(0.25);
  WeightedFmeasureLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 2e-2, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
*/

}  // namespace caffe
