#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/graph_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class Graph4CMakeModularLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Graph4CMakeModularLayerTest() : 
    bottom_(new Blob<Dtype>(2, 8, 5, 5)),
    top_(new Blob<Dtype>(2, 8, 5, 5)) {

    FillerParameter filler_param;
    filler_param.set_min(0.0);
    filler_param.set_max(1.0);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(bottom_);

    blob_bottom_vec_.push_back(bottom_);

    blob_top_vec_.push_back(top_);
  }
  virtual ~Graph4CMakeModularLayerTest() {
    delete bottom_;
    delete top_;
  }

  Blob<Dtype>* const bottom_;
  Blob<Dtype>* const top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(Graph4CMakeModularLayerTest, TestDtypesAndDevices);


TYPED_TEST(Graph4CMakeModularLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Graph4CMakeModularLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 2e-2, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}


}  // namespace caffe
