
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/graph_layers.hpp"
#include "caffe/util/math_functions.hpp"

// channel layout 
#define UD_CHANNEL 0
#define LR_CHANNEL 4
#define E00_OFFSET 0
#define E10_OFFSET 1
#define E01_OFFSET 2
#define E11_OFFSET 3

namespace caffe {


template <typename Dtype>
void Graph4CMakeModularLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->channels(), 8) <<
      "Graph4CMakeModular layer input must have 8 channels";

  vector<int> shape;
  shape.push_back(bottom[0]->num());
  shape.push_back(bottom[0]->channels());
  shape.push_back(bottom[0]->height());
  shape.push_back(bottom[0]->width());
  top[0]->Reshape(shape);
}

template <typename Dtype>
Dtype Graph4CMakeModularLayer<Dtype>::min(Dtype x, Dtype y, Dtype z) {
  return std::min(std::min(x,y), z);
}

template <typename Dtype>
Dtype Graph4CMakeModularLayer<Dtype>::buddy_min(Dtype x, Dtype y, Dtype z, 
	Dtype x_bud, Dtype y_bud, Dtype z_bud) {
  return x < y ? (x < z ? x_bud : z_bud) : (y < z ? y_bud : z_bud);
}

/*
	Blob 0 - pairwise energies
		channels 0-3 are for up/down energies
			value at [h,w] is the energy between pixels [h,w] and [h+1,w]
			Thus the last row is not used
		channels 4-7 are for left/right energies
			value at [h,w] is the energy between pixels [h,w] and [h,w+1]
			Thus the last col is not used
		see #defines

	Need to enforce modularity, i.e. E(1,0) + E(0,1) >= E(1,1) + E(0,0)
		Set E(0,0) = min(E(1,0), E(0,1), E(0,0))
		Set E(1,1) = min(E(1,0), E(0,1), E(1,1))
		Set E(1,0) = E(1,0)
		Set E(0,1) = E(0,1)
*/
template <typename Dtype>
void Graph4CMakeModularLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const int num = bottom[0]->num(); 
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int spatial_size = height * width;

  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  caffe_set(top[0]->count(), (Dtype) 0., top_data);

  for (int n = 0; n < num; n++) {
	for (int h = 0; h < height; h++) {
	  for (int w = 0; w < width; w++) {
	  	int num_offset = 8 * n * spatial_size;
		int spatial_offset = h * height + w;
		
		// UD data
		Dtype e_00 = bottom_data[num_offset + ((UD_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset];
		Dtype e_10 = bottom_data[num_offset + ((UD_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset];
		Dtype e_01 = bottom_data[num_offset + ((UD_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset];
		Dtype e_11 = bottom_data[num_offset + ((UD_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset];
        top_data[num_offset + ((UD_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset] = min(e_00, e_10, e_01);
        top_data[num_offset + ((UD_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset] = min(e_11, e_10, e_01);
        top_data[num_offset + ((UD_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset] = e_10;
        top_data[num_offset + ((UD_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset] = e_01;

		// LR data
		e_00 = bottom_data[num_offset + ((LR_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset];
		e_10 = bottom_data[num_offset + ((LR_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset];
		e_01 = bottom_data[num_offset + ((LR_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset];
		e_11 = bottom_data[num_offset + ((LR_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset];
        top_data[num_offset + ((LR_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset] = min(e_00, e_10, e_01);
        top_data[num_offset + ((LR_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset] = min(e_11, e_10, e_01);
        top_data[num_offset + ((LR_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset] = e_10;
        top_data[num_offset + ((LR_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset] = e_01;
	  }
	} // end height
  } // end num
}


template <typename Dtype>
void Graph4CMakeModularLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int num = bottom[0]->num(); 
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const int spatial_size = height * width;

    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    for (int n = 0; n < num; n++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int num_offset = 8 * n * spatial_size;
      	  int spatial_offset = h * height + w;

		  // UD data
		  Dtype e_00 = bottom_data[num_offset + ((UD_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset];
		  Dtype e_10 = bottom_data[num_offset + ((UD_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset];
		  Dtype e_01 = bottom_data[num_offset + ((UD_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset];
		  Dtype e_11 = bottom_data[num_offset + ((UD_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset];

		  Dtype e_00_diff = top_diff[num_offset + ((UD_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset];
		  Dtype e_10_diff = top_diff[num_offset + ((UD_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset];
		  Dtype e_01_diff = top_diff[num_offset + ((UD_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset];
		  Dtype e_11_diff = top_diff[num_offset + ((UD_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset];

          bottom_diff[num_offset + ((UD_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset] = 
		  	buddy_min(e_00, e_10, e_01, e_00_diff, (Dtype) 0, (Dtype) 0);
          bottom_diff[num_offset + ((UD_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset] = 
		  	buddy_min(e_11, e_10, e_01, e_11_diff, (Dtype) 0, (Dtype) 0);
          bottom_diff[num_offset + ((UD_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset] = e_10_diff + 
		  	buddy_min(e_00, e_10, e_01, (Dtype) 0, e_00_diff, (Dtype) 0) +
		  	buddy_min(e_11, e_10, e_01, (Dtype) 0, e_11_diff, (Dtype) 0);
          bottom_diff[num_offset + ((UD_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset] = e_01_diff + 
		  	buddy_min(e_00, e_10, e_01, (Dtype) 0, (Dtype) 0, e_00_diff) +
		  	buddy_min(e_11, e_10, e_01, (Dtype) 0, (Dtype) 0, e_11_diff);

		  // LR data
		  e_00 = bottom_data[num_offset + ((LR_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset];
		  e_10 = bottom_data[num_offset + ((LR_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset];
		  e_01 = bottom_data[num_offset + ((LR_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset];
		  e_11 = bottom_data[num_offset + ((LR_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset];

		  e_00_diff = top_diff[num_offset + ((LR_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset];
		  e_10_diff = top_diff[num_offset + ((LR_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset];
		  e_01_diff = top_diff[num_offset + ((LR_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset];
		  e_11_diff = top_diff[num_offset + ((LR_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset];

          bottom_diff[num_offset + ((LR_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset] = 
		  	buddy_min(e_00, e_10, e_01, e_00_diff, (Dtype) 0, (Dtype) 0);
          bottom_diff[num_offset + ((LR_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset] = 
		  	buddy_min(e_11, e_10, e_01, e_11_diff, (Dtype) 0, (Dtype) 0);
          bottom_diff[num_offset + ((LR_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset] = e_10_diff + 
		  	buddy_min(e_00, e_10, e_01, (Dtype) 0, e_00_diff, (Dtype) 0) +
		  	buddy_min(e_11, e_10, e_01, (Dtype) 0, e_11_diff, (Dtype) 0);
          bottom_diff[num_offset + ((LR_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset] = e_01_diff + 
		  	buddy_min(e_00, e_10, e_01, (Dtype) 0, (Dtype) 0, e_00_diff) +
		  	buddy_min(e_11, e_10, e_01, (Dtype) 0, (Dtype) 0, e_11_diff);
        }
      } // end height
    } // end num
  }
}


#ifdef CPU_ONLY
STUB_GPU(Graph4CMakeModularLayer);
#endif

INSTANTIATE_CLASS(Graph4CMakeModularLayer);
REGISTER_LAYER_CLASS(Graph4CMakeModular);

}  // namespace caffe

#undef UD_CHANNEL
#undef LR_CHANNEL
#undef E00_OFFSET
#undef E10_OFFSET
#undef E01_OFFSET
#undef E11_OFFSET
