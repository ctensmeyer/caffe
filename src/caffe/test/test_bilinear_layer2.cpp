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
class BilinearInterpolationLayerTest2 : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BilinearInterpolationLayerTest2()
      : blob_bottom_data_(new Blob<Dtype>(2, 2, 12, 10)),
        blob_bottom_dims_(new Blob<Dtype>(1, 1, 1, 2)),
        blob_top_data_(new Blob<Dtype>(2, 2, 19, 7)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-2.0);
    filler_param.set_max(2.0);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

	Dtype* dims = blob_bottom_dims_->mutable_cpu_data();
	dims[0] = 19; // height
	dims[1] = 7; // width

	blob_bottom_vec_.push_back(blob_bottom_data_);
	blob_bottom_vec_.push_back(blob_bottom_dims_);
	blob_top_vec_.push_back(blob_top_data_);

  }
  virtual ~BilinearInterpolationLayerTest2() {
    delete blob_bottom_data_;
    delete blob_bottom_dims_;
    delete blob_top_data_;
  }


  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_dims_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

/*
TYPED_TEST_CASE(BilinearInterpolationLayerTest2, TestDtypesAndDevices);

TYPED_TEST(BilinearInterpolationLayerTest2, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BilinearInterpolationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1702);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
*/

}  // namespace caffe
