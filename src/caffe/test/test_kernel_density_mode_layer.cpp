#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/kernel_density_mode_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class KernelDensityModeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  KernelDensityModeLayerTest() : 
    input_(new Blob<Dtype>(1, 1, 3, 3)),
    output_(new Blob<Dtype>(1, 2, 1, 1)) {

	Dtype* input = input_->mutable_cpu_data();

	input[0] = 4; input[1] = 2; input[2] = 2;
	input[3] = 3; input[4] = 0; input[5] = 1;
	input[6] = 5; input[7] = 5; input[8] = 8;

    blob_bottom_vec_.push_back(input_);

    blob_top_vec_.push_back(output_);
  }
  virtual ~KernelDensityModeLayerTest() {
    delete input_;
    delete output_;
  }

  void TestForward() {
    typedef typename TypeParam::Dtype Dtype;
    const double kErrorMargin = 1e-4;
	double radii[] = {0.5, 1.75, 2, 2.5, 3};
	double dh, dw;

	for (int k = 0; k < 5; k++) {
      LayerParameter layer_param;
      layer_param.mutable_kernel_param()->set_radius(radii[k]);
      KernelDensityModeLayer<Dtype> layer(layer_param);

      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      double h = output_->cpu_data()[0];
      double w = output_->cpu_data()[1];
	  double val = layer.GridVal(input_->cpu_data(), 3, 3, h, w);

	  double eps = 0.000001;
	  double h_lower = layer.GridVal(input_->cpu_data(), 3, 3, h - eps, w);
	  double h_upper = layer.GridVal(input_->cpu_data(), 3, 3, h + eps, w);
	  double finite_diff_left_dh = (val - h_lower) / eps;
	  double finite_diff_right_dh = (h_upper - val) / eps;
	  double finite_diff_diff_dh = std::abs(finite_diff_left_dh - finite_diff_right_dh);

	  double w_lower = layer.GridVal(input_->cpu_data(), 3, 3, h, w - eps);
	  double w_upper = layer.GridVal(input_->cpu_data(), 3, 3, h, w + eps);
	  double finite_diff_left_dw = (val - w_lower) / eps;
	  double finite_diff_right_dw = (w_upper - val) / eps;
	  double finite_diff_diff_dw = std::abs(finite_diff_left_dw - finite_diff_right_dw);

      // discontinuties occur with linear kernel on top of grid points
      if (finite_diff_diff_dh < kErrorMargin && finite_diff_diff_dw < kErrorMargin) {
	    layer.dGridVal(input_->cpu_data(), 3, 3, h, w, dh, dw);
        EXPECT_NEAR(dh, 0., kErrorMargin) << "  radius: " << radii[k] << " h,w: (" << h << "," << w << ")";
        EXPECT_NEAR(dw, 0., kErrorMargin) << "  radius: " << radii[k] << " h,w: (" << h << "," << w << ")";
	  } else {
	    LOG(INFO) << "Skipping radius " << radii[k] << " due to derivative discontinuity at (" << h << "," << w << ")";
	    LOG(INFO) << "   dh left/right/diff: " << finite_diff_left_dh << "," << finite_diff_right_dh << "," << finite_diff_diff_dh;
	    LOG(INFO) << "   dw left/right/diff: " << finite_diff_left_dw << "," << finite_diff_right_dw << "," << finite_diff_diff_dw;
	  }
	}
  }

  void TestOverallGrad() {
    typedef typename TypeParam::Dtype Dtype;
    const double kErrorMargin = 1e-4;

    LayerParameter layer_param;
    layer_param.mutable_kernel_param()->set_radius(2);
    layer_param.mutable_kernel_param()->set_grad_thresh(10e-10);
    KernelDensityModeLayer<Dtype> layer(layer_param);

    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    double h_star = output_->cpu_data()[0];
    double w_star = output_->cpu_data()[1];

	double dh = 0;
	double dw = 0;
	layer.dGridVal(input_->cpu_data(), 3, 3, h_star, w_star, dh, dw);

	LOG(INFO) << "h*,w* = (" << h_star << "," << w_star << ")";
	LOG(INFO) << "dh,dw = (" << dh << "," << dw << ")";

	double eps = 0.001;
	Dtype* input = input_->mutable_cpu_data();
	for (int h = 0; h < 3; h++) {
      for (int w = 0; w < 3; w++) {
		Dtype grid_orig = input[3*h+w];
	    LOG(INFO) << " ";
		LOG(INFO) << "h,w = (" << h << "," << w << ")";

		input[3*h+w] = grid_orig - eps;
        layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        double h_star_lower = output_->cpu_data()[0];
        double w_star_lower = output_->cpu_data()[1];

		input[3*h+w] = grid_orig + eps;
        layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		input[3*h+w] = grid_orig;
        double h_star_upper = output_->cpu_data()[0];
        double w_star_upper = output_->cpu_data()[1];

		double finite_diff_h_star = (h_star_upper - h_star_lower) / (2 * eps);
		double finite_diff_w_star = (w_star_upper - w_star_lower) / (2 * eps);

/*
		double finite_diff_left_h_star = (h_star - h_star_lower) / eps;
		double finite_diff_left_w_star = (w_star - w_star_lower) / eps;

		double finite_diff_right_h_star = (h_star_upper - h_star) / eps;
		double finite_diff_right_w_star = (w_star_upper - w_star) / eps;
*/

		double dh_star_dg = 0;
		double dw_star_dg = 0;

		// failed attempt at analytical gradient
		//layer.dOutdGrid(h_star, w_star, h, w, dh_star_dg, dw_star_dg);

		// layer version of finite diff
		layer.FiniteDiff(input, 3, 3, eps, h_star, w_star, h, w, dh_star_dg, dw_star_dg);

	    LOG(INFO) << "    upper h*,w* = (" << h_star_upper << "," << w_star_upper << ")";
	    LOG(INFO) << "    lower h*,w* = (" << h_star_lower << "," << w_star_lower << ")";
	    LOG(INFO) << "    layer dh,dw = (" << dh_star_dg   << "," << dw_star_dg   << ")";
	    LOG(INFO) << "    test  dh,dw  = (" << finite_diff_h_star  << "," << finite_diff_w_star << ")";
		/*
	    LOG(INFO) << "    left dh,dw  = (" << finite_diff_left_h_star  << "," << finite_diff_left_w_star  << ")";
	    LOG(INFO) << "    right dh,dw = (" << finite_diff_right_h_star << "," << finite_diff_right_w_star << ")";
	    LOG(INFO) << "    ratio dh,dw = (" << dh_star_dg / finite_diff_h_star << "," << dw_star_dg / finite_diff_w_star << ")";
		*/

        EXPECT_NEAR(dh_star_dg, finite_diff_h_star, kErrorMargin) << " ";
        EXPECT_NEAR(dw_star_dg, finite_diff_w_star, kErrorMargin) << " ";
	  }
	}

  }

  void TestGridGrad() {
    typedef typename TypeParam::Dtype Dtype;
    const double kErrorMargin = 1e-3;
    LayerParameter layer_param;
    layer_param.mutable_kernel_param()->set_radius(2);
    KernelDensityModeLayer<Dtype> layer(layer_param);

	double min = 0.01;
	double max = 3.0;
	double step = 0.1;
	int num_steps = (int) std::ceil((max - min) / step);
	double eps = 10e-5;
	//double min_eps = 0.0000001;
	double dh, dw;
	double finite_diff_both_dh, finite_diff_both_dw;
	double finite_diff_left_dh, finite_diff_left_dw;
	double finite_diff_right_dh, finite_diff_right_dw;

	for (int h_step = 0; h_step < num_steps; h_step++) {
	  double h = h_step * step + min;
	  for (int w_step = 0; w_step < num_steps; w_step++) {
	    double w = w_step * step + min;
		double val = layer.GridVal(input_->cpu_data(), 3, 3, h, w);
	    layer.dGridVal(input_->cpu_data(), 3, 3, h, w, dh, dw);

        //for (eps = 0.001; eps >= min_eps; eps /= 10) {
		  double h_lower = layer.GridVal(input_->cpu_data(), 3, 3, h - eps, w);
		  double h_upper = layer.GridVal(input_->cpu_data(), 3, 3, h + eps, w);
		  finite_diff_both_dh = (h_upper - h_lower) / (2 * eps);
		  finite_diff_left_dh = (val - h_lower) / eps;
		  finite_diff_right_dh = (h_upper - val) / eps;
		  //LOG(INFO) << "dh: (" << finite_diff_both_dh << "," << finite_diff_left_dh << "," << finite_diff_right_dh << ")";
		//}

        //for (eps = 0.001; eps >= min_eps; eps /= 10) {
		  double w_lower = layer.GridVal(input_->cpu_data(), 3, 3, h, w - eps);
		  double w_upper = layer.GridVal(input_->cpu_data(), 3, 3, h, w + eps);
		  finite_diff_both_dw = (w_upper - w_lower) / (2 * eps);
		  finite_diff_left_dw = (val - w_lower) / eps;
		  finite_diff_right_dw = (w_upper - val) / eps;
		  //LOG(INFO) << "dw: (" << finite_diff_both_dw << "," << finite_diff_left_dw << "," << finite_diff_right_dw << ")";
		//}
		//LOG(INFO) << "(h,w) = (" << h << "," << w << ")\n";

		double finite_diff_diff_dh = std::abs(finite_diff_left_dh - finite_diff_right_dh);
		if (finite_diff_diff_dh < kErrorMargin) {
          EXPECT_NEAR(dh, finite_diff_both_dh, kErrorMargin) << "    at (h,w) = (" << h << "," << w << ")\n" 
		  	<< "    dh:(" << finite_diff_both_dh << "," << finite_diff_left_dh << "," << finite_diff_right_dh << ")";
		} else {
		  LOG(INFO) << "Skipping (" << h << "," << w << ") dh:(" << finite_diff_both_dh << "," << 
		  				finite_diff_left_dh << "," << finite_diff_right_dh << ")";
		}

		double finite_diff_diff_dw = std::abs(finite_diff_left_dw - finite_diff_right_dw);
		if (finite_diff_diff_dw < kErrorMargin) {
          EXPECT_NEAR(dw, finite_diff_both_dw, kErrorMargin) << "    at (h,w) = (" << h << "," << w << ")\n"
		  	<< "    dw:(" << finite_diff_both_dw << "," << finite_diff_left_dw << "," << finite_diff_right_dw << ")";
		} else {
		  LOG(INFO) << "Skipping (" << h << "," << w << ") dw:(" << finite_diff_both_dw << "," << 
		  				finite_diff_left_dw << "," << finite_diff_right_dw << ")";
		}

        //EXPECT_NEAR(finite_diff_left_dh, finite_diff_right_dh, kErrorMargin) << "at (h,w) = (" << h << "," << w << ")\n";
        //EXPECT_NEAR(finite_diff_left_dw, finite_diff_right_dw, kErrorMargin) << "at (h,w) = (" << h << "," << w << ")\n";

		LOG(INFO) << "";
	  }
	}
  }

  void TestLinearKernel() {
    typedef typename TypeParam::Dtype Dtype;
    const double kErrorMargin = 1e-5;
    LayerParameter layer_param;
    layer_param.mutable_kernel_param()->set_radius(2);
    KernelDensityModeLayer<Dtype> layer(layer_param);

    // r = 2

	// 4 2 2
	// 3 0 1
	// 5 5 8
	
	// at h=0.5 w=1.75, val = 6.614601, dv/dh = 5.042369, dv/dw = -3.987002

	double expected_vals[] =    {6.5, 
	                            6.171573,
						        3.5, 
						        9.550253, 
						        11.06497, 
						        8.050253, 
						        9., 
						        12.67157, 
						        11.};
	double expected_dh_vals[] = {4., 
	                            3.914214,
								4.5, 
								1.560660,
								3.974874,
								4.060660,
								-3.5, 
							   -2.414214, 
								-1.5};
	double expected_dw_vals[] = {2., 
	                           -1.707106, 
								-3, 
								2.974874, 
								-0.64644, 
								-3.97487, 
								6.5, 
								0.792893, 
								-5.};

    double val, dh, dw;
	for (int h = 0; h < 3; h++) {
      for (int w = 0; w < 3; w++) {
	    val = layer.GridVal(input_->cpu_data(), 3, 3, h, w);
        EXPECT_NEAR(val, expected_vals[h*3+w], kErrorMargin) << "\n";

		dh = 0;
		dw = 0;
	    layer.dGridVal(input_->cpu_data(), 3, 3, h, w, dh, dw);
        EXPECT_NEAR(dh, expected_dh_vals[h*3+w], kErrorMargin) << "\n";
        EXPECT_NEAR(dw, expected_dw_vals[h*3+w], kErrorMargin) << "\n";
	  }
	}
	val = layer.GridVal(input_->cpu_data(), 3, 3, 0.5, 1.75);
	EXPECT_NEAR(val, 6.614601, kErrorMargin) << "\n";

	layer.dGridVal(input_->cpu_data(), 3, 3, 0.5, 1.75, dh, dw);
	EXPECT_NEAR(dh, 5.042369, kErrorMargin) << "\n";
	EXPECT_NEAR(dw, -3.987002, kErrorMargin) << "\n";

  }

  Blob<Dtype>* const input_;
  Blob<Dtype>* const output_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

//TYPED_TEST_CASE(KernelDensityModeLayerTest, TestDtypesAndDevices);
TYPED_TEST_CASE(KernelDensityModeLayerTest, ::testing::Types<CPUDevice<float> >);



TYPED_TEST(KernelDensityModeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_kernel_param()->set_radius(2);
  KernelDensityModeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 4e-2, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}


/*
TYPED_TEST(KernelDensityModeLayerTest, TestForward) {
  this->TestForward();
}
*/

/*
TYPED_TEST(KernelDensityModeLayerTest, TestOverallGrad) {
  this->TestOverallGrad();
}
*/

/*
TYPED_TEST(KernelDensityModeLayerTest, TestGridGrad) {
  this->TestGridGrad();
}

TYPED_TEST(KernelDensityModeLayerTest, TestLinearKernel) {
  this->TestLinearKernel();
}
*/


}  // namespace caffe
