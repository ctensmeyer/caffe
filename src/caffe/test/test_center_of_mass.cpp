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
    top_(new Blob<Dtype>(2, 2, 2, 1)) {

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
  }
  virtual ~CenterOfMassLayerTest() {
    delete bottom_;
    delete top_;
  }

  Blob<Dtype>* const bottom_;
  Blob<Dtype>* const top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(CenterOfMassLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(CenterOfMassLayerTest, ::testing::Types<CPUDevice<float> >);


TYPED_TEST(CenterOfMassLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CenterOfMassLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}


}  // namespace caffe
