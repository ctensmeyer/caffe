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
class Graph4CEnergyLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Graph4CEnergyLayerTest() : 
    unary_(new Blob<Dtype>(1, 2, 2, 2)),
    pair_(new Blob<Dtype>(1, 8, 2, 2)),
    labels_(new Blob<Dtype>(1, 1, 2, 2)),
    energy_(new Blob<Dtype>(1, 1, 1, 1)) {

	Dtype* unary = unary_->mutable_cpu_data();
	// background unaries
	unary[0] = 1; unary[1] = 5;
	unary[2] = 8; unary[3] = 3;

	// foreground unaries
	unary[4] = 3; unary[5] = 6;
	unary[6] = 7; unary[7] = 2;

    blob_bottom_vec_.push_back(unary_);

	Dtype* pair = pair_->mutable_cpu_data();
	// UD 0,0 energy
	pair[0] = 1; pair[1] = 2;
	pair[2] = 0; pair[3] = 0;

	// UD 1,0 energy
	pair[4] = 7; pair[5] = 2;
	pair[6] = 0; pair[7] = 0;

	// UD 0,1 energy
	pair[8] = 4; pair[9] = 1;
	pair[10] = 0; pair[11] = 0;

	// UD 1,1 energy
	pair[12] = 3; pair[13] = 1;
	pair[14] = 0; pair[15] = 0;

	// LR 0,0 energy
	pair[16] = 3; pair[17] = 0;
	pair[18] = 1; pair[19] = 0;

	// LR 1,0 energy
	pair[20] = 9; pair[21] = 0;
	pair[22] = 4; pair[23] = 0;

	// LR 0,1 energy
	pair[24] = 5; pair[25] = 0;
	pair[26] = 2; pair[27] = 0;

	// LR 1,1 energy
	pair[28] = 3; pair[29] = 0;
	pair[30] = 1; pair[31] = 0;

    blob_bottom_vec_.push_back(pair_);

	Dtype* labels = labels_->mutable_cpu_data();
	labels[0] = 1;
	labels[1] = 0;
	labels[2] = 1;
	labels[3] = 0;

    blob_bottom_vec_.push_back(labels_);

    blob_top_vec_.push_back(energy_);
  }
  virtual ~Graph4CEnergyLayerTest() {
    delete unary_;
    delete pair_;
    delete labels_;
    delete energy_;
  }

  Blob<Dtype>* const unary_;
  Blob<Dtype>* const pair_;
  Blob<Dtype>* const labels_;
  Blob<Dtype>* const energy_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(Graph4CEnergyLayerTest, TestDtypesAndDevices);


TYPED_TEST(Graph4CEnergyLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Graph4CEnergyLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 2e-2, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}


}  // namespace caffe
