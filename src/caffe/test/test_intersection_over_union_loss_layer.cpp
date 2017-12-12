#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/intersection_over_union_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class IntersectionOverUnionLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  IntersectionOverUnionLossLayerTest() : 
    input_(new Blob<Dtype>(1, 1, 1, 1)),
    gt_(new Blob<Dtype>(1, 1, 1, 1)),
    output_(new Blob<Dtype>(1, 1, 1, 1)) {

    blob_bottom_vec_.push_back(input_);
    blob_bottom_vec_.push_back(gt_);

	blob_bottom_vec2_.push_back(gt_);
	blob_bottom_vec2_.push_back(input_);
    blob_top_vec_.push_back(output_);
  }
  virtual ~IntersectionOverUnionLossLayerTest() {
    delete input_;
    delete gt_;
    delete output_;
  }

  void TestTwoTrianglesForward() {
    typedef typename TypeParam::Dtype Dtype;
    const double kErrorMargin = 1e-4;

	input_->Reshape(1, 3, 2, 1);
	gt_->Reshape(1, 3, 2, 1);

	Dtype* input = input_->mutable_cpu_data();

	// (0,0), (10, 0), (5, 10)
	// area = 50
	input[0] = 0; input[1] = 0;
	input[2] = 5; input[3] = 10;
	input[4] = 10; input[5] = 0;

	// translated 5 units up
	// (0,5), (10, 5), (5, 15)
	// area = 50
	Dtype* gt = gt_->mutable_cpu_data();
	gt[0] = 0; gt[1] = 5;
	gt[2] = 5; gt[3] = 15;
	gt[4] = 10; gt[5] = 5;

    LayerParameter layer_param;
	IntersectionOverUnionLossLayer<Dtype> layer(layer_param);
	layer.SetUp(blob_bottom_vec_, blob_top_vec_);
	layer.Reshape(blob_bottom_vec_, blob_top_vec_);
	layer.Forward(blob_bottom_vec_, blob_top_vec_);

	Dtype out = output_->cpu_data()[0];
	Dtype iou = 12.5 / (100 - 12.5);
    EXPECT_NEAR(out, 1 - iou, kErrorMargin); 
  }

  void TestTwoTrianglesBackward() {
    typedef typename TypeParam::Dtype Dtype;
    const double kErrorMargin = 1e-4;
    const double eps = 1e-3;

	input_->Reshape(1, 3, 2, 1);
	gt_->Reshape(1, 3, 2, 1);

	Dtype* input = input_->mutable_cpu_data();

	// (0,0), (5, 10), (10, 0)
	// area = 50
	input[0] = 0; input[1] = 0;
	input[2] = 5; input[3] = 10;
	input[4] = 10; input[5] = 0;

	// translated 5 units up
	// (0,5), (5, 15), (10, 5)
	// area = 50
	Dtype* gt = gt_->mutable_cpu_data();
	gt[0] = 0; gt[1] = 5;
	gt[2] = 5; gt[3] = 15;
	gt[4] = 10; gt[5] = 5;

    LayerParameter layer_param;
	IntersectionOverUnionLossLayer<Dtype> layer(layer_param);

    GradientChecker<Dtype> checker(eps, kErrorMargin, 1701, 1, 0.01);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec2_,
        this->blob_top_vec_, 0);
  }

  void TestMultipleIntersectingPolygons() {
    typedef typename TypeParam::Dtype Dtype;
    const double kErrorMargin = 1e-4;
    const double eps = 1e-3;

	input_->Reshape(1, 4, 2, 1);
	gt_->Reshape(1, 3, 2, 1);

	Dtype* input = input_->mutable_cpu_data();

	// non-convex (two horns)
	// (0, 10) (10, 20) (5, 10) (10, 0)
	input[0] = 0;  input[1] = 10;
	input[2] = 10; input[3] = 20;
	input[4] = 5;  input[5] = 10;
	input[6] = 10; input[7] = 0;

	// triangle should clip both horns
	// (5,0), (10, 25), (30, -1)
	Dtype* gt = gt_->mutable_cpu_data();
	gt[0] = 5;  gt[1] = 0;
	gt[2] = 10; gt[3] = 25;
	gt[4] = 30; gt[5] = -1;

    LayerParameter layer_param;
	IntersectionOverUnionLossLayer<Dtype> layer(layer_param);

    GradientChecker<Dtype> checker(eps, kErrorMargin, 1701, 1, 0.01);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
	/*
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec2_,
        this->blob_top_vec_, 0);
	*/
  }

  void TestBatch() {
    typedef typename TypeParam::Dtype Dtype;
    const double kErrorMargin = 1e-4;
    const double eps = 1e-3;

	input_->Reshape(2, 4, 2, 1);
	gt_->Reshape(2, 3, 2, 1);

	Dtype* input = input_->mutable_cpu_data();

	// non-convex (two horns)
	// (0, 10) (10, 20) (5, 10) (10, 0)
	input[0] = 0;  input[1] = 10;
	input[2] = 10; input[3] = 20;
	input[4] = 5;  input[5] = 10;
	input[6] = 10; input[7] = 0;

    // second (perturbed) polygon
	input[8] = 1;   input[9]  = 9;
	input[10] = 11; input[11] = 21;
	input[12] = 6;  input[13] = 11;
	input[14] = 11; input[15] = 2;

	// triangle should clip both horns
	// (5,0), (10, 25), (30, -1)
	Dtype* gt = gt_->mutable_cpu_data();
	gt[0] = 5;  gt[1] = 0;
	gt[2] = 10; gt[3] = 25;
	gt[4] = 30; gt[5] = -1;

    // second (perturbed) polygon
	gt[6] = 6;  gt[7] = 2;
	gt[8] = 8; gt[9] = 20;
	gt[10] = 33; gt[11] = -2;

    LayerParameter layer_param;
	IntersectionOverUnionLossLayer<Dtype> layer(layer_param);

    GradientChecker<Dtype> checker(eps, kErrorMargin, 1701, 1, 0.01);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
	/*
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec2_,
        this->blob_top_vec_, 0);
	*/
  }


  Blob<Dtype>* const input_;
  Blob<Dtype>* const gt_;
  Blob<Dtype>* const output_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec2_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(IntersectionOverUnionLossLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(IntersectionOverUnionLossLayerTest, ::testing::Types<CPUDevice<double> >);

TYPED_TEST(IntersectionOverUnionLossLayerTest, TestTwoTrianglesForward) {
  this->TestTwoTrianglesForward();
}

TYPED_TEST(IntersectionOverUnionLossLayerTest, TestTwoTrianglesBackward) {
  this->TestTwoTrianglesBackward();
}

TYPED_TEST(IntersectionOverUnionLossLayerTest, TestMultipleIntersectingPolygons) {
  this->TestMultipleIntersectingPolygons();
}

TYPED_TEST(IntersectionOverUnionLossLayerTest, TestBatch) {
  this->TestBatch();
}

}  // namespace caffe
