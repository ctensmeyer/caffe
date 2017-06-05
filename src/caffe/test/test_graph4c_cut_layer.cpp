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
class Graph4CCutLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Graph4CCutLayerTest() : 
    bottom_(new Blob<Dtype>(3, 4, 2, 2)),
    top1_(new Blob<Dtype>(3, 1, 2, 2)),
    top2_(new Blob<Dtype>(3, 1, 1, 1)) {

	Dtype* bottom = bottom_->mutable_cpu_data();

	// instance 0
	// source edges
	bottom[0] = 7; bottom[1] = 2;
	bottom[2] = 0; bottom[3] = 0;

	// term edges
	bottom[4] = 0; bottom[5] = 0;
	bottom[6] = 8; bottom[7] = 3;

	// ud edges
	bottom[8]  = 2; bottom[9]  = 4;
	bottom[10] = 0; bottom[11] = 0;

	// lr edges
	bottom[12] = 1; bottom[13] = 0;
	bottom[14] = 2; bottom[15] = 0;

	// instance 1
	// source edges
	bottom[16] = 2; bottom[17] = 4;
	bottom[18] = 10; bottom[19] = 0;

	// term edges
	bottom[20] = 0; bottom[21] = 0;
	bottom[22] = 0; bottom[23] = 11;

	// ud edges
	bottom[24] = 2; bottom[25] = 1;
	bottom[26] = 0; bottom[27] = 0;

	// lr edges
	bottom[28] = 3; bottom[29] = 0;
	bottom[30] = 4; bottom[31] = 0;

	// instance 2
	// source edges
	bottom[32] = 10; bottom[33] = 0;
	bottom[34] = 0; bottom[35] = 17;

	// term edges
	bottom[36] = 0; bottom[37] = 48;
	bottom[38] = 20; bottom[39] = 0;

	// ud edges
	bottom[40] = 2; bottom[41] = 10;
	bottom[42] = 0; bottom[43] = 0;

	// lr edges
	bottom[44] = 11; bottom[45] = 0;
	bottom[46] = 8; bottom[47] = 0;


    blob_bottom_vec_.push_back(bottom_);

    blob_top_vec_.push_back(top1_);
    blob_top_vec_.push_back(top2_);
  }
  virtual ~Graph4CCutLayerTest() {
    delete bottom_;
    delete top1_;
    delete top2_;
  }

  Blob<Dtype>* const bottom_;
  Blob<Dtype>* const top1_;
  Blob<Dtype>* const top2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

#ifdef USE_GRAPHCUT
  void TestCut() {
    const Dtype kErrorMargin = 1e-5;
    LayerParameter layer_param;
    Graph4CCutLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

	const Dtype* cut_cost = top2_->cpu_data();
    EXPECT_NEAR(cut_cost[0], (Dtype)8, kErrorMargin);
    EXPECT_NEAR(cut_cost[1], (Dtype)10, kErrorMargin);
    EXPECT_NEAR(cut_cost[2], (Dtype)27, kErrorMargin);

	const Dtype* cut = top1_->cpu_data();

	// instance 0
    EXPECT_NEAR(cut[0], (Dtype)0, kErrorMargin);
    EXPECT_NEAR(cut[1], (Dtype)1, kErrorMargin);
    EXPECT_NEAR(cut[2], (Dtype)1, kErrorMargin);
    EXPECT_NEAR(cut[3], (Dtype)1, kErrorMargin);

	// instance 1
    EXPECT_NEAR(cut[4], (Dtype)0, kErrorMargin);
    EXPECT_NEAR(cut[5], (Dtype)0, kErrorMargin);
    EXPECT_NEAR(cut[6], (Dtype)0, kErrorMargin);
    EXPECT_NEAR(cut[7], (Dtype)1, kErrorMargin);

	// instance 2
    EXPECT_NEAR(cut[8], (Dtype)1, kErrorMargin);
    EXPECT_NEAR(cut[9], (Dtype)1, kErrorMargin);
    EXPECT_NEAR(cut[10], (Dtype)1, kErrorMargin);
    EXPECT_NEAR(cut[11], (Dtype)1, kErrorMargin);
  }
#endif
};

#ifdef USE_GRAPHCUT
TYPED_TEST_CASE(Graph4CCutLayerTest, TestDtypesAndDevices);


TYPED_TEST(Graph4CCutLayerTest, TestCut) {
  this->TestCut();
}
#endif


}  // namespace caffe
