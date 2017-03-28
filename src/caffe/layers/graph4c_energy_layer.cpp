
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/graph_layers.hpp"
#include "caffe/util/math_functions.hpp"

// 3 bottom blobs
#define UNARY_BLOB_IDX 0
#define PAIR_BLOB_IDX 1
#define LABEL_BLOB_IDX 2

// channel layout of PAIR blob
#define UD_CHANNEL 0
#define LR_CHANNEL 4
#define E00_OFFSET 0
#define E10_OFFSET 1
#define E01_OFFSET 2
#define E11_OFFSET 3

namespace caffe {


template <typename Dtype>
void Graph4CEnergyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[UNARY_BLOB_IDX]->num(), bottom[PAIR_BLOB_IDX]->num()) <<
      "Graph4CEnergy layer inputs must have the same number of instances.";
  CHECK_EQ(bottom[UNARY_BLOB_IDX]->num(), bottom[LABEL_BLOB_IDX]->num()) <<
      "Graph4CEnergy layer inputs must have the same number of instances.";

  CHECK_EQ(bottom[UNARY_BLOB_IDX]->channels(), 2) <<
      "Graph4CEnergy layer unary input must have 2 channels";
  CHECK_EQ(bottom[PAIR_BLOB_IDX]->channels(), 8) <<
      "Graph4CEnergy layer pair input must have 8 channels";
  CHECK_EQ(bottom[LABEL_BLOB_IDX]->channels(), 1) <<
      "Graph4CEnergy layer label input must have 1 channel";

  CHECK_EQ(bottom[UNARY_BLOB_IDX]->height(), bottom[PAIR_BLOB_IDX]->height()) <<
      "Graph4CEnergy layer inputs must have the same height.";
  CHECK_EQ(bottom[UNARY_BLOB_IDX]->height(), bottom[LABEL_BLOB_IDX]->height()) <<
      "Graph4CEnergy layer inputs must have the same height.";

  CHECK_EQ(bottom[UNARY_BLOB_IDX]->width(), bottom[PAIR_BLOB_IDX]->width()) <<
      "Graph4CEnergy layer inputs must have the same width.";
  CHECK_EQ(bottom[UNARY_BLOB_IDX]->width(), bottom[LABEL_BLOB_IDX]->width()) <<
      "Graph4CEnergy layer inputs must have the same width.";

  vector<int> shape;
  shape.push_back(bottom[0]->num());
  shape.push_back(1);
  top[0]->Reshape(shape);
}

/*
 3 bottom blobs
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
	Blob 2 - labeling
		Binary values
 1 top blob of size num x 1
   The scalar energies associated with each labeling
*/
template <typename Dtype>
void Graph4CEnergyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const int num = bottom[UNARY_BLOB_IDX]->num(); 
  const int height = bottom[UNARY_BLOB_IDX]->height();
  const int width = bottom[UNARY_BLOB_IDX]->width();
  const int spatial_size = height * width;

  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* unary_energy = bottom[UNARY_BLOB_IDX]->cpu_data();
  const Dtype* pair_energy = bottom[PAIR_BLOB_IDX]->cpu_data();
  const Dtype* labels = bottom[LABEL_BLOB_IDX]->cpu_data();

  caffe_set(top[0]->count(), (Dtype) 0., top_data);

  for (int n = 0; n < num; n++) {
	Dtype energy = 0;
	for (int h = 0; h < height; h++) {
	  for (int w = 0; w < width; w++) {
	  	int unary_num_offset = 2 * n * spatial_size;
	  	int pair_num_offset = 8 * n * spatial_size;
	  	int label_num_offset = n * spatial_size;
		int spatial_offset = h * height + w;

		Dtype cur_label = labels[label_num_offset + spatial_offset];

		if (cur_label) {
		  // foreground unary energy
		  energy += unary_energy[unary_num_offset + spatial_size + spatial_offset];
		} else {
		  // background unary energy
		  energy += unary_energy[unary_num_offset + spatial_offset];
		}

		if (h + 1 < height) {
		  // not bottom row
		  spatial_offset = (h + 1) * height + w;
		  Dtype neb_label = labels[label_num_offset + spatial_offset];
		  int channel = UD_CHANNEL;
		  if (cur_label) {
		    if (neb_label) {
			  channel += E11_OFFSET;
			} else {
			  channel += E10_OFFSET;
			}
		  } else {
		    if (neb_label) {
			  channel += E01_OFFSET;
			} else {
			  channel += E00_OFFSET;
			}
		  }
		  spatial_offset = h * height + w;
		  energy += pair_energy[pair_num_offset + (channel * spatial_size) + spatial_offset];
		}

		if (w + 1 < width) {
		  // not right col
		  spatial_offset = h * height + w + 1;
		  Dtype neb_label = labels[label_num_offset + spatial_offset];
		  int channel = LR_CHANNEL;
		  if (cur_label) {
		    if (neb_label) {
			  channel += E11_OFFSET;
			} else {
			  channel += E10_OFFSET;
			}
		  } else {
		    if (neb_label) {
			  channel += E01_OFFSET;
			} else {
			  channel += E00_OFFSET;
			}
		  }
		  spatial_offset = h * height + w;
		  energy += pair_energy[pair_num_offset + (channel * spatial_size) + spatial_offset];
		}
	  }
	} // end height
	top_data[n] = energy / spatial_size;
  } // end num
}


template <typename Dtype>
void Graph4CEnergyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) {
    DLOG(INFO) << "Graph4CEnergyLayer does not propagate down to blob index 2";
  }
  if (propagate_down[0] or propagate_down[1]) {
    const int num = bottom[UNARY_BLOB_IDX]->num(); 
    const int height = bottom[UNARY_BLOB_IDX]->height();
    const int width = bottom[UNARY_BLOB_IDX]->width();
    const int spatial_size = height * width;

    Dtype* top_diff = top[0]->mutable_cpu_diff();

    Dtype* unary_diff = bottom[UNARY_BLOB_IDX]->mutable_cpu_diff();
    Dtype* pair_diff = bottom[PAIR_BLOB_IDX]->mutable_cpu_diff();
    const Dtype* labels = bottom[LABEL_BLOB_IDX]->cpu_data();

    for (int n = 0; n < num; n++) {
	  // dE/de = 1, where e is any individual energy param
	  // thus dLoss/de = 1 * dLoss/dE = top_diff[n]
	  Dtype diff = top_diff[n] / spatial_size; 
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int unary_num_offset = 2 * n * spatial_size;
          int pair_num_offset = 8 * n * spatial_size;
          int label_num_offset = n * spatial_size;
      	  int spatial_offset = h * height + w;

      	  Dtype cur_label = labels[label_num_offset + spatial_offset];

      	  if (cur_label) {
      	    // foreground unary energy
      	    unary_diff[unary_num_offset + spatial_size + spatial_offset] = diff;
      	  } else {
      	    // background unary energy
      	    unary_diff[unary_num_offset + spatial_offset] = diff;
      	  }

      	  if (h + 1 < height) {
      	    // not bottom row
      	    spatial_offset = (h + 1) * height + w;
      	    Dtype neb_label = labels[label_num_offset + spatial_offset];
      	    int channel = UD_CHANNEL;
      	    if (cur_label) {
      	      if (neb_label) {
      	  	    channel += E11_OFFSET;
      	  	  } else {
      	  	    channel += E10_OFFSET;
      	  	  }
      	    } else {
      	      if (neb_label) {
      	  	    channel += E01_OFFSET;
      	  	  } else {
      	  	    channel += E00_OFFSET;
      	  	  }
      	    }
      	    spatial_offset = h * height + w;
      	    pair_diff[pair_num_offset + (channel * spatial_size) + spatial_offset] = diff;
      	  }

      	  if (w + 1 < width) {
      	    // not right col
      	    spatial_offset = h * height + w + 1;
      	    Dtype neb_label = labels[label_num_offset + spatial_offset];
      	    int channel = LR_CHANNEL;
      	    if (cur_label) {
      	      if (neb_label) {
      	  	    channel += E11_OFFSET;
      	  	  } else {
      	  	    channel += E10_OFFSET;
      	  	  }
      	    } else {
      	      if (neb_label) {
      	  	    channel += E01_OFFSET;
      	  	  } else {
      	  	    channel += E00_OFFSET;
      	  	  }
      	    }
      	    spatial_offset = h * height + w;
      	    pair_diff[pair_num_offset + (channel * spatial_size) + spatial_offset] = diff;
      	  }
        }
      } // end height
    } // end num
  }
}


#ifdef CPU_ONLY
STUB_GPU(Graph4CEnergyLayer);
#endif

INSTANTIATE_CLASS(Graph4CEnergyLayer);
REGISTER_LAYER_CLASS(Graph4CEnergy);

}  // namespace caffe

#undef UNARY_BLOB_IDX
#undef PAIR_BLOB_IDX
#undef LABEL_BLOB_IDX
#undef UD_CHANNEL
#undef LR_CHANNEL
#undef E00_OFFSET
#undef E10_OFFSET
#undef E01_OFFSET
#undef E11_OFFSET
