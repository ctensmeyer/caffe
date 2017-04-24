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
class Graph4CEnergyLayerTest3 : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Graph4CEnergyLayerTest3() : 
    bottom1_(new Blob<Dtype>(1, 2, 2, 2)),
    bottom2_(new Blob<Dtype>(1, 8, 2, 2)),
    bottom3_(new Blob<Dtype>(1, 1, 2, 2)),
    top1_(new Blob<Dtype>(1, 1, 1, 1)),
    expected_energy_(new Blob<Dtype>(16, 1, 1, 1)) {

	// order wrt graphcut.py is x1, x2, x4, x3
	Dtype* bottom1 = bottom1_->mutable_cpu_data();

	Dtype not_used = 0;  // can be any arbitrary value

	// energy 0 unary terms
	bottom1[0] = 10; bottom1[1] = 3;  // x1 and x2
	bottom1[2] = 20; bottom1[3] = 1;  // x4 and x3

	// energy 1 unary terms
	bottom1[4] =  5; bottom1[5] = 2;  // x1 and x2
	bottom1[6] =  2; bottom1[7] = 8;  // x4 and x3

    blob_bottom_vec_.push_back(bottom1_);

	Dtype* bottom2 = bottom2_->mutable_cpu_data();

	// UD energies E00
	bottom2[0] = 2; bottom2[1] = 8; // x1-x4, x2-x3
	bottom2[2] = not_used; bottom2[3] = not_used; // bottom row not used

	// UD energies E10 - top node is 1, bottom node is 0
	bottom2[4] = 8; bottom2[5] = 6; // x1-x4, x2-x3
	bottom2[6] = not_used; bottom2[7] = not_used; // bottom row not used
	
	// UD energies E01 - top node is 0, bottom node is 1
	bottom2[8] =  0; bottom2[9] = 20; // x1-x4, x2-x3
	bottom2[10] = not_used; bottom2[11] = not_used; // bottom row not used

	// UD energies E11
	bottom2[12] = 4; bottom2[13] = 8; // x1-x4, x2-x3
	bottom2[14] = not_used; bottom2[15] = not_used; // bottom row not used

	// LR energies E00
	bottom2[16] = 16; bottom2[17] = not_used; // x1-x2, not used
	bottom2[18] =  4; bottom2[19] = not_used; // x4-x3, not used

	// LR energies E10 - left node is 1, right node is 0
	bottom2[20] = 24; bottom2[21] = not_used; // x1-x2, not used
	bottom2[22] = 10; bottom2[23] = not_used; // x4-x3, not used

	// LR energies E01 - left node is 0, right node is 1
	bottom2[24] = 5; bottom2[25] = not_used; // x1-x2, not used
	bottom2[26] = 6; bottom2[27] = not_used; // x4-x3, not used

	// LR energies E11 
	bottom2[28] = 2; bottom2[29] = not_used; // x1-x2, not used
	bottom2[30] = 4; bottom2[31] = not_used; // x4-x3, not used

    blob_bottom_vec_.push_back(bottom2_);
    blob_bottom_vec_.push_back(bottom3_);

    blob_top_vec_.push_back(top1_);

	Dtype* expected_energy = expected_energy_->mutable_cpu_data();
	Dtype norm = 4.;  // energy is normalized by the spatial size
	expected_energy[0] = 94 / norm; // 0000
	expected_energy[1] = 84 / norm; // 0001
	expected_energy[2] = 129 / norm; // 0010
	expected_energy[3] = 103 / norm; // 0011
	expected_energy[4] = 67 / norm; // 0100
	expected_energy[5] = 57 / norm; // 0101
	expected_energy[6] = 82 / norm; // 0110
	expected_energy[7] = 56 / norm; // 0111
	expected_energy[8] = 117 / norm;  // 1000
	expected_energy[9] = 103 / norm;  // 1001
	expected_energy[10] = 152 / norm; // 1010 
	expected_energy[11] = 122 / norm; // 1011 
	expected_energy[12] = 68 / norm; // 1100 
	expected_energy[13] = 54 / norm; // 1101 
	expected_energy[14] = 83 / norm; // 1110 
	expected_energy[15] = 53 / norm; // 1111 
  }
  virtual ~Graph4CEnergyLayerTest3() {
    delete bottom1_;
    delete bottom2_;
    delete bottom3_;
    delete expected_energy_;
    delete top1_;
  }

  Blob<Dtype>* const bottom1_;
  Blob<Dtype>* const bottom2_;
  Blob<Dtype>* const bottom3_;
  Blob<Dtype>* const top1_;
  Blob<Dtype>* const expected_energy_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestEnergy() {
    const Dtype kErrorMargin = 1e-5;
    LayerParameter layer_param;
    Graph4CEnergyLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);

	const Dtype* expected_energy = expected_energy_->cpu_data();
	Dtype* bottom3 = bottom3_->mutable_cpu_data();

    int idx = 0;
	for (int x1 = 0; x1 <= 1; x1++) {
	  bottom3[0] = x1;
	  for (int x2 = 0; x2 <= 1; x2++) {
	    bottom3[1] = x2;
	    for (int x3 = 0; x3 <= 1; x3++) {
	      bottom3[3] = x3;
	      for (int x4 = 0; x4 <= 1; x4++) {
	        bottom3[2] = x4;
            layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
			Dtype energy = top1_->cpu_data()[0];
            EXPECT_NEAR(energy, expected_energy[idx], kErrorMargin);
			idx += 1;
		  }
		}
	  }
	}
  }
};

TYPED_TEST_CASE(Graph4CEnergyLayerTest3, TestDtypesAndDevices);


TYPED_TEST(Graph4CEnergyLayerTest3, TestConstruct) {
  this->TestEnergy();
}


}  // namespace caffe
