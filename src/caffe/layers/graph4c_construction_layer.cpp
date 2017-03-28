
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/graph_layers.hpp"
#include "caffe/util/math_functions.hpp"

// 2 bottom blobs
#define UNARY_BLOB_IDX 0
#define PAIR_BLOB_IDX 1

// channel layout of PAIR blob
#define UD_CHANNEL 0
#define LR_CHANNEL 4
#define E00_OFFSET 0
#define E10_OFFSET 1
#define E01_OFFSET 2
#define E11_OFFSET 3

// top blob channels
#define SOURCE_OFFSET 0
#define TERM_OFFSET 1
#define UD_EDGE_OFFSET 2
#define LR_EDGE_OFFSET 3

namespace caffe {


template <typename Dtype>
void Graph4CConstructorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[UNARY_BLOB_IDX]->num(), bottom[PAIR_BLOB_IDX]->num()) <<
      "Graph4CConstructor layer inputs must have the same number of instances.";

  CHECK_EQ(bottom[UNARY_BLOB_IDX]->channels(), 2) <<
      "Graph4CConstructor layer unary input must have 2 channels";
  CHECK_EQ(bottom[PAIR_BLOB_IDX]->channels(), 8) <<
      "Graph4CConstructor layer pairwise input must have 8 channels";

  CHECK_EQ(bottom[UNARY_BLOB_IDX]->height(), bottom[PAIR_BLOB_IDX]->height()) <<
      "Graph4CConstructor layer inputs must have the same height.";

  CHECK_EQ(bottom[UNARY_BLOB_IDX]->width(), bottom[PAIR_BLOB_IDX]->width()) <<
      "Graph4CConstructor layer inputs must have the same width.";

  vector<int> shape;
  shape.push_back(bottom[UNARY_BLOB_IDX]->num());
  shape.push_back(4);
  shape.push_back(bottom[UNARY_BLOB_IDX]->height());
  shape.push_back(bottom[UNARY_BLOB_IDX]->width());

  top[0]->Reshape(shape);
}

/*
 2 bottom blobs, 1 for the unary energies, 1 for the label pair energy terms
 	Blob 0 - unary energies
		channel 0 is for background (class 0) unary energies
		channel 1 is for foreground (class 1) unary energies
	Blob 1 - pairwise energies
		channels 0-3 are for up/down energies
			value at [h,w] is the energy between pixels [h,w] and [h+1,w]
			Thus the last row is not used
		channels 4-7 are for left/right energies
			value at [h,w] is the energy between pixels [h,w] and [h,w+1]
			Thus the last col is not used
		see #defines
 1 top blob of size num x 4 x height x width.
   Channel 0 is the edge weights from the source node (background)
   Channel 1 is the edge weights to the terminal node (foreground)
   Channels 2-3 are edge weights between up-down and left-right nodes
      Note that these cases are not distringuished here, so it is up
	  to the energy computing layer to decide which is which.
	  Precisely, Channel 2 corresponds to the input channels 0-3
	             Channel 3 corresponds to the input channels 4-7
*/
template <typename Dtype>
void Graph4CConstructorLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const int num = top[0]->num(); 
  const int height = top[0]->height();
  const int width = top[0]->width();
  const int spatial_size = height * width;

  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* unary_energy = bottom[UNARY_BLOB_IDX]->cpu_data();
  const Dtype* pair_energy = bottom[PAIR_BLOB_IDX]->cpu_data();

  caffe_set(top[0]->count(), (Dtype) 0., top_data);

  for (int n = 0; n < num; n++) {
	for (int h = 0; h < height; h++) {
	  for (int w = 0; w < width; w++) {
		// compute source/terminal edge weights
		// edge weights are handled interspersed as well

	  	int unary_num_offset = 2 * n * spatial_size;
	  	int pair_num_offset = 8 * n * spatial_size;
	  	int top_num_offset = 4 * n * spatial_size;
		int spatial_offset = h * height + w;

	    // c_i = unary_energy_i[1] - unary_energy_i[0];
	    Dtype c_i = (unary_energy[unary_num_offset + spatial_size + spatial_offset] - 
		             unary_energy[unary_num_offset + spatial_offset]);

		// c_i += \sum_j pair_energy_ij[1,0] + [1,1] - [0,1] - [0,0]
		// also computes the edge weights
		if (h - 1 >= 0) {
		  // up neighbor
		  spatial_offset = (h-1) * height + w;
		  c_i += (pair_energy[pair_num_offset + ((UD_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset] +
		          pair_energy[pair_num_offset + ((UD_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset] -
		          pair_energy[pair_num_offset + ((UD_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset] -
		          pair_energy[pair_num_offset + ((UD_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset]);
	    }
		if (h + 1 < height) {
		  // down neighbor
		  spatial_offset = h * height + w;
		  c_i += (pair_energy[pair_num_offset + ((UD_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset] +
		          pair_energy[pair_num_offset + ((UD_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset] -
		          pair_energy[pair_num_offset + ((UD_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset] -
		          pair_energy[pair_num_offset + ((UD_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset]);

	      // UD edge weight
		  Dtype ud_edge_weight = 
	             (pair_energy[pair_num_offset + ((UD_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset] +
			      pair_energy[pair_num_offset + ((UD_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset] -
			      pair_energy[pair_num_offset + ((UD_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset] -
			      pair_energy[pair_num_offset + ((UD_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset]);
		  top_data[top_num_offset + (UD_EDGE_OFFSET * spatial_size) + spatial_offset] = ud_edge_weight;
		}
		if (w - 1 >= 0) {
		  // left neighbor
		  spatial_offset = h * height + w - 1;
		  c_i += (pair_energy[pair_num_offset + ((LR_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset] +
		          pair_energy[pair_num_offset + ((LR_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset] -
		          pair_energy[pair_num_offset + ((LR_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset] -
		          pair_energy[pair_num_offset + ((LR_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset]);
		}
		if (w + 1 < width) {
		  // right neighbor
		  spatial_offset = h * height + w;
		  c_i += (pair_energy[pair_num_offset + ((LR_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset] +
		          pair_energy[pair_num_offset + ((LR_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset] -
		          pair_energy[pair_num_offset + ((LR_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset] -
		          pair_energy[pair_num_offset + ((LR_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset]);

	      // LR edge weight
		  Dtype lr_edge_weight = 
	             (pair_energy[pair_num_offset + ((UD_CHANNEL + E10_OFFSET) * spatial_size) + spatial_offset] +
			      pair_energy[pair_num_offset + ((UD_CHANNEL + E01_OFFSET) * spatial_size) + spatial_offset] -
			      pair_energy[pair_num_offset + ((UD_CHANNEL + E11_OFFSET) * spatial_size) + spatial_offset] -
			      pair_energy[pair_num_offset + ((UD_CHANNEL + E00_OFFSET) * spatial_size) + spatial_offset]);
		  top_data[top_num_offset + (LR_EDGE_OFFSET * spatial_size) + spatial_offset] = lr_edge_weight;
		}
	    spatial_offset = h * height + w;

	    // these are preset to 0
	    if (c_i >= 0) {
	      // connect to source (background class)
	      top_data[top_num_offset + (SOURCE_OFFSET * spatial_size) + spatial_offset] = c_i;
	    } else {
	      // connect to terminal (foreground class)
	      top_data[top_num_offset +   (TERM_OFFSET * spatial_size) + spatial_offset] = -1 * c_i;
	    }
	  }
	}
  }
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(Graph4CConstructorLayer);
#endif

INSTANTIATE_CLASS(Graph4CConstructorLayer);
REGISTER_LAYER_CLASS(Graph4CConstructor);

}  // namespace caffe

#undef UNARY_BLOB_IDX
#undef PAIR_BLOB_IDX
#undef UD_CHANNEL
#undef LR_CHANNEL
#undef E00_OFFSET
#undef E10_OFFSET
#undef E01_OFFSET
#undef E11_OFFSET
#undef SOURCE_OFFSET
#undef TERM_OFFSET
#undef UD_EDGE_OFFSET
#undef LR_EDGE_OFFSET

