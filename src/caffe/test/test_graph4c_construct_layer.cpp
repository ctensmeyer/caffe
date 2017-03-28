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

/*
This unit test is based on a particular energy function over 4 variables
with square connectivity.
The specification of the energy function is this python code:

def energy(x1, x2, x3, x4):
	e = unary_x1(x1) + unary_x2(x2) + unary_x3(x3) + unary_x4(x4)
	e += pair_1_2(x1, x2)
	e += pair_1_3(x1, x3)
	e += pair_1_4(x1, x4)

	e += pair_2_1(x2, x1)
	e += pair_2_3(x2, x3)
	e += pair_2_4(x2, x4)

	e += pair_3_1(x3, x1)
	e += pair_3_2(x3, x2)
	e += pair_3_4(x3, x4)

	e += pair_4_1(x4, x1)
	e += pair_4_2(x4, x2)
	e += pair_4_3(x4, x3)
	return e

def unary_x1(x1):
    return 5 if x1 else 10

def unary_x2(x2):
    return 2 if x2 else 3

def unary_x3(x3):
    return 8 if x3 else 1

def unary_x4(x4):
    return 2 if x4 else 20

def pair_1_2(x1, x2):
    return (2 * x1 * x2 +
            24 * x1 * (1 - x2) +
            5 * (1 - x1) * x2 +
            16 * (1 - x1) * (1 - x2))

def pair_1_3(x1, x3):
    return 0

def pair_1_4(x1, x4):
    return (4 * x1 * x4 +
            8 * x1 * (1 - x4) +
            0 * (1 - x1) * x4 +
            2 * (1 - x1) * (1 - x4))

def pair_2_1(x2, x1):
    return pair_1_2(x1, x2)

def pair_2_3(x2, x3):
    return (8 * x2 * x3 +
            6 * x2 * (1 - x3) +
            20 * (1 - x2) * x3 +
            8 * (1 - x2) * (1 - x3))

def pair_2_4(x2, x4):
    return 0

def pair_3_1(x3, x1):
    return 0

def pair_3_2(x3, x2):
    return pair_2_3(x2, x3)

def pair_3_4(x3, x4):
    return (4 * x3 * x4 +
            6 * x3 * (1 - x4) +
            10 * (1 - x3) * x4 +
            4 * (1 - x3) * (1 - x4))

def pair_4_1(x4, x1):
    return pair_1_4(x1, x4)

def pair_4_2(x4, x2):
    return 0

def pair_4_3(x4, x3):
    return pair_3_4(x3, x4)


It should create the following graph:

    -10*---S-----
    |           |
    v           |
    x1->11<-x2--|---
    |        |  |   |
    v        v  |   |
    2       10  17* 48
    ^        ^  |   |
    |        |  |   |
    x4-->8<-x3<--   |
    |               | 
    |               | 
    20----->T<------
    
Where * indicates edges that are part of the minimum cut for
a total cut cost of 27

*/

template <typename TypeParam>
class Graph4CConstructLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Graph4CConstructLayerTest() : 
    bottom1_(new Blob<Dtype>(1, 2, 2, 2)),
    bottom2_(new Blob<Dtype>(1, 8, 2, 2)),
    top1_(new Blob<Dtype>(1, 4, 2, 2)) {

	// order wrt graphcut.py is x1, x2, x4, x3
	Dtype* bottom1 = bottom1_->mutable_cpu_data();

	// energy 0 unary terms
	bottom1[0] = 10; bottom1[1] = 3;  // x1 and x2
	bottom1[2] = 20; bottom1[3] = 1;  // x4 and x3

	// energy 1 unary terms
	bottom1[0] =  5; bottom1[1] = 2;  // x1 and x2
	bottom1[2] =  2; bottom1[3] = 8;  // x4 and x3

    blob_bottom_vec_.push_back(bottom1_);

	Dtype* bottom2 = bottom2_->mutable_cpu_data();

	// UD energies E00
	bottom2[0] = 2; bottom2[1] = 8; // x1-x4, x2-x3
	bottom2[2] = 0; bottom2[3] = 0; // bottom row not used

	// UD energies E10 - top node is 1, bottom node is 0
	bottom2[4] = 8; bottom2[5] = 6; // x1-x4, x2-x3
	bottom2[6] = 0; bottom2[7] = 0; // bottom row not used
	
	// UD energies E01 - top node is 0, bottom node is 1
	bottom2[8] =  0; bottom2[9] = 20; // x1-x4, x2-x3
	bottom2[10] = 0; bottom2[11] = 0; // bottom row not used

	// UD energies E11
	bottom2[12] = 4; bottom2[13] = 8; // x1-x4, x2-x3
	bottom2[14] = 0; bottom2[15] = 0; // bottom row not used

	// LR energies E00
	bottom2[16] = 16; bottom2[17] = 0; // x1-x2, not used
	bottom2[18] =  4; bottom2[19] = 0; // x4-x3, not used

	// LR energies E10 - left node is 1, right node is 0
	bottom2[20] = 24; bottom2[21] = 0; // x1-x2, not used
	bottom2[22] = 10; bottom2[23] = 0; // x4-x3, not used

	// LR energies E01 - left node is 0, right node is 1
	bottom2[20] = 5; bottom2[21] = 0; // x1-x2, not used
	bottom2[22] = 6; bottom2[23] = 0; // x4-x3, not used

	// LR energies E11 
	bottom2[20] = 2; bottom2[21] = 0; // x1-x2, not used
	bottom2[22] = 4; bottom2[23] = 0; // x4-x3, not used

    blob_bottom_vec_.push_back(bottom2_);

    blob_top_vec_.push_back(top1_);
  }
  virtual ~Graph4CConstructLayerTest() {
    delete bottom1_;
    delete bottom2_;
    delete top1_;
  }

  Blob<Dtype>* const bottom1_;
  Blob<Dtype>* const bottom2_;
  Blob<Dtype>* const top1_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestConstruct() {
    const Dtype kErrorMargin = 1e-5;
    LayerParameter layer_param;
    Graph4CConstructorLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);


	const Dtype* graph_weights = top1_->cpu_data();
    EXPECT_NEAR(graph_weights[0], (Dtype)10, kErrorMargin); // x1-s
    EXPECT_NEAR(graph_weights[1], (Dtype)0, kErrorMargin); // x2-s
    EXPECT_NEAR(graph_weights[2], (Dtype)0, kErrorMargin); // x4-s
    EXPECT_NEAR(graph_weights[3], (Dtype)17, kErrorMargin); // x3-s
    EXPECT_NEAR(graph_weights[4], (Dtype)0, kErrorMargin); // x1-t
    EXPECT_NEAR(graph_weights[5], (Dtype)48, kErrorMargin); // x2-t
    EXPECT_NEAR(graph_weights[6], (Dtype)20, kErrorMargin); // x4-t
    EXPECT_NEAR(graph_weights[7], (Dtype)0, kErrorMargin); // x3-t
    EXPECT_NEAR(graph_weights[8], (Dtype)2, kErrorMargin); // x1-x4
    EXPECT_NEAR(graph_weights[9], (Dtype)10, kErrorMargin); // x2-x3
    //EXPECT_NEAR(graph_weights[10], (Dtype)8, kErrorMargin); // not used
    //EXPECT_NEAR(graph_weights[11], (Dtype)8, kErrorMargin); // not used
    EXPECT_NEAR(graph_weights[12], (Dtype)11, kErrorMargin); // x1-x2
    EXPECT_NEAR(graph_weights[13], (Dtype)8, kErrorMargin); // x4-x3
    //EXPECT_NEAR(graph_weights[14], (Dtype)8, kErrorMargin); // not used
    //EXPECT_NEAR(graph_weights[15], (Dtype)8, kErrorMargin); // not used

  }
};

TYPED_TEST_CASE(Graph4CConstructLayerTest, TestDtypesAndDevices);


TYPED_TEST(Graph4CConstructLayerTest, TestConstruct) {
  this->TestConstruct();
}


}  // namespace caffe
