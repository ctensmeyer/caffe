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
class RelativeDarknessLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RelativeDarknessLayerTest() : 
	blob_bottom_data_(new Blob<Dtype>(1, 1, 10, 10)),
	blob_top_data_(new Blob<Dtype>(1, 3, 10, 10)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

	//Dtype* bottom = blob_bottom_data_->mutable_cpu_data();
	//bottom[0] = 0; //bottom[1] = 1; bottom[2] = 2; bottom[3] = 3;

    blob_bottom_vec_.push_back(blob_bottom_data_);

    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~RelativeDarknessLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_data_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RelativeDarknessLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(RelativeDarknessLayerTest, ::testing::Types<CPUDevice<float> >);



TYPED_TEST(RelativeDarknessLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_relative_darkness_param()->set_kernel_size(5);
  layer_param.mutable_relative_darkness_param()->set_min_param_value(0.01);
  layer_param.mutable_relative_darkness_param()->mutable_filler()->set_type("uniform");
  layer_param.mutable_relative_darkness_param()->mutable_filler()->set_min(0.01);
  RelativeDarknessLayer<Dtype> layer(layer_param);
  layer.LayerSetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<bool> prop_down;
  prop_down.push_back(false);
  layer.Backward(this->blob_bottom_vec_, prop_down, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-5, 1e-1, 1701, 1, 0.01);
  // only check parameter blob
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 100); 
  //checker.CheckGradientSingle(&layer, this->blob_bottom_vec_,
  //    this->blob_top_vec_, 100, 0, 1); 
}

}  // namespace caffe
