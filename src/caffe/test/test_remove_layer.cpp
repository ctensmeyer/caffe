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
class RemoveLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RemoveLayerTest() : 
  	    blob_bottom_data_(new Blob<Dtype>(5, 4, 3, 5)),
        blob_bottom_index_(new Blob<Dtype>(5, 1, 1, 1)),
        blob_top_(new Blob<Dtype>(5, 3, 3, 5)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0.0);
    filler_param.set_max(1.0);
    UniformFiller<Dtype> filler(filler_param);

    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);

    // Assume binary targets
    int count = blob_bottom_index_->count();
	Dtype* target = blob_bottom_index_->mutable_cpu_data();
    for (int i = 0; i < count; i++) {
	  target[i] = i % 4;
	}

    blob_bottom_vec_.push_back(blob_bottom_index_);

    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~RemoveLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_index_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_index_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RemoveLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(RemoveLayerTest, ::testing::Types<CPUDevice<float> >);


TYPED_TEST(RemoveLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RemoveLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
